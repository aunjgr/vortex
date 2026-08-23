// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::ops::Range;
use std::sync::Arc;

use futures::Stream;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::splits::Splits;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::TaskFuture;
use crate::scan::tasks::split_exec;

/// One lazily constructed scan task and the source row range it covers.
pub struct ScanTask<A: 'static + Send> {
    row_range: Range<u64>,
    future: TaskFuture<Option<A>>,
}

impl<A: 'static + Send> ScanTask<A> {
    /// Returns the source row range covered by this task.
    pub fn row_range(&self) -> &Range<u64> {
        &self.row_range
    }

    /// Consumes this task and returns its execution future.
    pub fn into_future(self) -> TaskFuture<Option<A>> {
        self.future
    }
}

/// Demand-driven iterator over scan tasks.
///
/// Calling [`Iterator::next`] is the point at which projection I/O is registered for a split.
/// This lets callers bound scan lookahead without first allocating every split future.
pub struct ScanTasks<A: 'static + Send> {
    ranges: std::vec::IntoIter<Range<u64>>,
    selection: Selection,
    limit: Option<u64>,
    ctx: Arc<TaskContext<A>>,
    finished: bool,
}

impl<A: 'static + Send> Iterator for ScanTasks<A> {
    type Item = VortexResult<ScanTask<A>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        for range in self.ranges.by_ref() {
            let row_mask = self.selection.row_mask(&range);
            if row_mask.mask().all_false() {
                continue;
            }

            let row_range = row_mask.row_range();
            let future = match split_exec(Arc::clone(&self.ctx), row_mask, self.limit.as_mut()) {
                Ok(future) => future,
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            };
            if self.limit.is_some_and(|limit| limit == 0) {
                self.finished = true;
            }
            return Some(Ok(ScanTask { row_range, future }));
        }

        self.finished = true;
        None
    }
}

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan<A: 'static + Send> {
    session: VortexSession,
    layout_reader: LayoutReaderRef,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// The natural splits of the file.
    splits: Splits,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl RepeatedScan<ArrayRef> {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        row_range: Option<Range<u64>>,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        let iter = runtime.block_on_stream(stream);
        Ok(ArrayIteratorAdapter::new(dtype, iter))
    }

    pub fn execute_array_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }
}

impl<A: 'static + Send> RepeatedScan<A> {
    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub fn new(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
        limit: Option<u64>,
        dtype: DType,
    ) -> Self {
        Self {
            session,
            layout_reader,
            projection,
            filter,
            ordered,
            row_range,
            selection,
            splits,
            concurrency,
            map_fn,
            limit,
            dtype,
        }
    }

    /// Returns a demand-driven iterator over tasks intersecting `row_range`.
    pub fn tasks(&self, row_range: Option<Range<u64>>) -> VortexResult<ScanTasks<A>> {
        let ranges = self.task_ranges(row_range);
        let ctx = Arc::new(TaskContext {
            filter: self
                .filter
                .clone()
                .map(|filter| Arc::new(FilterExpr::new(filter))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
            mapper: Arc::clone(&self.map_fn),
        });

        Ok(ScanTasks {
            ranges: ranges.into_iter(),
            selection: self.selection.clone(),
            limit: self.limit,
            ctx,
            finished: false,
        })
    }

    /// Constructs all tasks intersecting `row_range`.
    ///
    /// Prefer [`Self::tasks`] when the caller can apply backpressure.
    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        self.tasks(row_range)?
            .map(|task| task.map(ScanTask::into_future))
            .collect()
    }

    fn task_ranges(&self, row_range: Option<Range<u64>>) -> Vec<Range<u64>> {
        let selection_range: Option<Range<u64>> = match &self.selection {
            Selection::IncludeByIndex(buf) if !buf.is_empty() => {
                Some(buf[0]..buf[buf.len() - 1] + 1)
            }
            Selection::IncludeRoaring(roaring) if !roaring.is_empty() => {
                Some(roaring.min().vortex_expect("empty")..roaring.max().vortex_expect("empty") + 1)
            }
            _ => None,
        };
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        let row_range = intersect_ranges(row_range.as_ref(), selection_range);

        match &self.splits {
            Splits::Natural(vec) => {
                debug_assert!(vec.is_sorted());
                let boundaries = match row_range {
                    None => vec.to_vec(),
                    Some(range) => {
                        if range.is_empty() {
                            return Vec::new();
                        }
                        let lo = vec.partition_point(|&x| x <= range.start);
                        let hi = vec.partition_point(|&x| x < range.end);
                        let mut boundaries = Vec::with_capacity(hi.saturating_sub(lo) + 2);
                        boundaries.push(range.start);
                        boundaries.extend_from_slice(&vec[lo..hi]);
                        boundaries.push(range.end);
                        boundaries
                    }
                };
                boundaries
                    .windows(2)
                    .map(|values| values[0]..values[1])
                    .collect()
            }
            Splits::Ranges(ranges) => match row_range {
                None => ranges.clone(),
                Some(range) => {
                    if range.is_empty() {
                        return Vec::new();
                    }
                    ranges
                        .iter()
                        .filter_map(move |candidate| {
                            let start = cmp::max(candidate.start, range.start);
                            let end = cmp::min(candidate.end, range.end);
                            (start < end).then_some(start..end)
                        })
                        .collect()
                }
            },
        }
    }

    pub fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        use futures::StreamExt;
        let num_workers = get_available_parallelism().unwrap_or(1);
        let concurrency = self.concurrency * num_workers;
        let handle = self.session.handle();

        use futures::FutureExt;
        let stream = futures::stream::iter(self.tasks(row_range)?).map(move |task| {
            let handle = handle.clone();
            async move { handle.spawn(task?.into_future()).await }.boxed()
        });

        let stream = if self.ordered {
            stream.buffered(concurrency).boxed()
        } else {
            stream.buffer_unordered(concurrency).boxed()
        };

        Ok(stream.filter_map(|chunk| async move { chunk.transpose() }))
    }
}

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (None, Some(r)) => Some(r),
        (Some(l), None) => Some(l.clone()),
        (Some(l), Some(r)) => Some(cmp::max(l.start, r.start)..cmp::min(l.end, r.end)),
    }
}
