// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::should_execute_dictionary_values;
use crate::ArrayRef;
use crate::CanonicalView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::execute::take_canonical;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeKernel;
use crate::scalar_fn::fns::like::LikeOptions;

impl LikeKernel for Dict {
    fn like(
        array: ArrayView<'_, Dict>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !should_execute_dictionary_values(array.values().len(), array.codes().len()) {
            return Ok(None);
        }
        let Some(pattern) = pattern.as_constant() else {
            return Ok(None);
        };

        let pattern = ConstantArray::new(pattern, array.values().len()).into_array();
        let matched_values = Like::try_new(array.values().clone(), pattern, options)?
            .into_array()
            .execute::<BoolArray>(ctx)?;
        let codes = array.codes().clone().execute::<PrimitiveArray>(ctx)?;

        Ok(Some(
            take_canonical(
                CanonicalView::Bool(matched_values.as_view()),
                codes.as_view(),
                ctx,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Bool;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    use crate::scalar_fn::fns::like::Like;
    use crate::scalar_fn::fns::like::LikeOptions;

    #[test]
    fn like_execute_dict_returns_canonical_bool() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinArray::from(vec!["hello", "world", "help"]).into_array(),
        )?
        .into_array();

        let pattern = ConstantArray::new("hello%", 4).into_array();
        let result = Like::try_new(dict, pattern, LikeOptions::default())?
            .into_array()
            .execute::<BoolArray>(&mut ctx)?
            .into_array();

        assert!(result.is::<Bool>());
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, false, true, false]),
            &mut ctx
        );
        Ok(())
    }
}
