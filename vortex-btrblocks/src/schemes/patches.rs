// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::patches::Patches;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::SchemeId;

/// Compresses the physical patch-location and patch-value children of a patched array.
pub fn compress_patched_children(
    compressor: &CascadingCompressor,
    array: Array<Patched>,
    compress_ctx: &CompressorContext,
    parent_id: SchemeId,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let slots = ArraySlots::from([
        Some(array.inner().clone()),
        Some(compressor.compress_child(array.lane_offsets(), compress_ctx, parent_id, 1, ctx)?),
        Some(compressor.compress_child(array.patch_indices(), compress_ctx, parent_id, 2, ctx)?),
        Some(compressor.compress_child(array.patch_values(), compress_ctx, parent_id, 3, ctx)?),
    ]);

    // SAFETY: each compressed child preserves the original logical dtype and values.
    unsafe { array.into_array().with_slots(slots) }
}

/// Compresses the given patches by downscaling integers and checking for constant values.
pub fn compress_patches(patches: Patches, ctx: &mut ExecutionCtx) -> VortexResult<Patches> {
    // Downscale the patch indices.
    let indices = patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .narrow(ctx)?
        .into_array();

    // Check if the values are constant.
    let values = patches.values();
    let values = if values
        .statistics()
        .compute_is_constant(ctx)
        .unwrap_or_default()
    {
        ConstantArray::new(values.execute_scalar(0, ctx)?, values.len()).into_array()
    } else {
        values.clone()
    };
    let chunk_offsets = patches
        .chunk_offsets()
        .as_ref()
        .map(|offsets| {
            let offsets_primitive = offsets
                .clone()
                .execute::<PrimitiveArray>(ctx)?
                .narrow(ctx)?
                .into_array();
            Ok::<ArrayRef, VortexError>(offsets_primitive)
        })
        .transpose()?;

    Patches::new(
        patches.array_len(),
        patches.offset(),
        indices,
        values,
        chunk_offsets,
    )
}
