// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::time::Duration;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::display::profile::DecompressionProfile;
use crate::display::profile::NodeTiming;
use crate::display::profile::ProfileOptions;

/// Extractor that adds a `throughput:` detail line from a measured [`DecompressionProfile`].
///
/// The line reports the time to canonicalize the subtree, its share of the whole tree's time, the
/// rates that time implies, and either the node's self time or the amount of child work it fuses
/// into itself.
pub struct ThroughputExtractor {
    profile: DecompressionProfile,
}

impl ThroughputExtractor {
    /// Annotate a tree with an already-measured profile.
    pub fn new(profile: DecompressionProfile) -> Self {
        Self { profile }
    }

    /// Measure `array` and annotate it with the result.
    pub fn measure(
        array: &ArrayRef,
        session: &VortexSession,
        options: ProfileOptions,
    ) -> VortexResult<Self> {
        Ok(Self::new(DecompressionProfile::measure(
            array, session, options,
        )?))
    }

    /// The profile backing this extractor.
    pub fn profile(&self) -> &DecompressionProfile {
        &self.profile
    }
}

impl TreeExtractor<ArrayRef, TreeContext> for ThroughputExtractor {
    fn write_details(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut crate::display::IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        let Some(timing) = self.profile.get(array) else {
            return Ok(());
        };
        let (indent, f) = f.parts();
        write!(
            f,
            "{indent}throughput: {}",
            Timing(timing, self.profile.root_time())
        )?;
        writeln!(f)
    }
}

struct Timing<'a>(&'a NodeTiming, Duration);

impl fmt::Display for Timing<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(timing, root) = *self;
        let percent = if root.is_zero() {
            0.0
        } else {
            100_f64 * timing.subtree.as_secs_f64() / root.as_secs_f64()
        };
        write!(
            f,
            "{} ({percent:.2}%) | in {} | out {} | {}",
            Elapsed(timing.subtree),
            Rate(timing.input_bytes_per_sec(), &["B", "kB", "MB", "GB"]),
            Rate(timing.output_bytes_per_sec(), &["B", "kB", "MB", "GB"]),
            Rate(timing.rows_per_sec(), &["row", "krow", "Mrow", "Grow"]),
        )?;
        match timing.fusion_saving() {
            Some(saving) => write!(f, " | fuses children (saves {})", Elapsed(saving)),
            None => write!(f, " | self {}", Elapsed(timing.self_time())),
        }
    }
}

/// A duration, rendered as `1.81ms`.
struct Elapsed(Duration);

impl fmt::Display for Elapsed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs_f64();
        for (scale, unit) in [(1.0, "s"), (1e-3, "ms"), (1e-6, "\u{b5}s")] {
            if secs >= scale {
                return write!(f, "{:.2}{unit}", secs / scale);
            }
        }
        write!(f, "{:.0}ns", secs * 1e9)
    }
}

/// A per-second rate, rendered in the largest unit that keeps it above one, e.g. `1.90 GB/s`.
struct Rate(f64, &'static [&'static str]);

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(rate, units) = *self;
        if !rate.is_finite() {
            return write!(f, "n/a");
        }
        let mut scale = 1.0;
        let mut unit = units[0];
        for next in &units[1..] {
            if rate < scale * 1e3 {
                break;
            }
            scale *= 1e3;
            unit = next;
        }
        write!(f, "{:.2} {unit}/s", rate / scale)
    }
}
