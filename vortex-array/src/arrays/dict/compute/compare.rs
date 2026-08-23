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
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

impl CompareKernel for Dict {
    fn compare(
        lhs: ArrayView<'_, Dict>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // if we have more values than codes, it is faster to canonicalise first.
        if !should_execute_dictionary_values(lhs.values().len(), lhs.codes().len()) {
            return Ok(None);
        }

        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(rhs) = rhs.as_constant() {
            let compare_result = lhs.values().clone().binary(
                ConstantArray::new(rhs, lhs.values().len()).into_array(),
                Operator::from(operator),
            )?;

            // Boolean dictionaries are not useful to consumers. Execute only the small values
            // result, then use the canonical boolean take kernel directly instead of publishing
            // a Dict<Bool> graph and sending it through the generic array executor.
            let compared_values = compare_result.execute::<BoolArray>(ctx)?;
            let codes = lhs.codes().clone().execute::<PrimitiveArray>(ctx)?;
            return Ok(Some(
                take_canonical(
                    CanonicalView::Bool(compared_values.as_view()),
                    codes.as_view(),
                    ctx,
                )?
                .into_array(),
            ));
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
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
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn constant_comparison_returns_canonical_bool() -> VortexResult<()> {
        let dictionary = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            buffer![10i32, 20, 30].into_array(),
        )?
        .into_array();
        let rhs = ConstantArray::new(20i32, dictionary.len()).into_array();
        let result = dictionary.binary(rhs, Operator::Lt)?;

        let mut ctx = array_session().create_execution_ctx();
        let result = result.execute::<BoolArray>(&mut ctx)?.into_array();
        assert!(result.is::<Bool>());
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, false, true, false]),
            &mut ctx
        );
        Ok(())
    }
}
