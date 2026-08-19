// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::Scope;
use crate::expr::Variable;
use crate::expr::VariableRef;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

/// Arrays supplied for variables in an outer expression scope.
///
/// Lambda parameters and captures are bound internally by their higher-order-function evaluator;
/// callers create this value only for variables supplied from outside the expression.
#[derive(Debug, Default)]
pub struct Bindings {
    root_dtype: Option<DType>,
    bindings: HashMap<VariableRef, ArrayRef>,
}

impl Bindings {
    /// Resolve named arrays against an outer expression scope.
    ///
    /// The scope may contain one external binding frame. Lambda frames are introduced and
    /// populated internally by the higher-order function that owns them.
    pub fn from_scope(
        scope: &Scope,
        bindings: impl IntoIterator<Item = (Variable, ArrayRef)>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            scope.depth() <= 1,
            "external expression bindings support one lexical binding frame"
        );

        let mut resolved = HashMap::new();
        for (variable, array) in bindings {
            let Some((dtype, variable_ref)) = scope.resolve(&variable) else {
                return Err(vortex_err!(
                    "binding supplied for unbound variable '{variable}'"
                ));
            };
            vortex_ensure!(
                array.dtype() == dtype,
                "binding for variable '{variable}' has dtype {}, expected {dtype}",
                array.dtype()
            );
            vortex_ensure!(
                resolved.insert(variable_ref, array).is_none(),
                "duplicate binding for variable '{variable}'"
            );
        }

        Ok(Self {
            root_dtype: Some(scope.root().clone()),
            bindings: resolved,
        })
    }

    fn check_root(&self, root: &ArrayRef) -> VortexResult<()> {
        if let Some(root_dtype) = &self.root_dtype {
            vortex_ensure!(
                root.dtype() == root_dtype,
                "expression scope dtype {root_dtype} does not match root array dtype {}",
                root.dtype()
            );
        }
        Ok(())
    }
}

/// Dynamic lexical state shared while an expression is applied to an array.
pub(crate) struct ApplyCtx {
    bindings: HashMap<VariableRef, ArrayRef>,
}

impl ApplyCtx {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    fn from_bindings(bindings: &Bindings) -> Self {
        Self {
            bindings: bindings.bindings.clone(),
        }
    }

    fn binding(&self, variable_ref: VariableRef) -> Option<&ArrayRef> {
        self.bindings.get(&variable_ref)
    }
}

impl ArrayRef {
    /// Apply a bound expression with no externally supplied variable values.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        apply(self, expr, &mut ApplyCtx::new())
    }

    /// Apply a bound expression with externally supplied variable values.
    pub fn apply_bound_with_bindings(
        self,
        expr: &BoundExpression,
        bindings: &Bindings,
    ) -> VortexResult<ArrayRef> {
        bindings.check_root(&self)?;
        apply(self, expr, &mut ApplyCtx::from_bindings(bindings))
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let scope = Scope::new(self.dtype().clone());
        let bound = expr.bind(&scope)?;
        self.apply_bound(&bound)
    }
}

/// Lower `expr` into an array in `root`'s row domain using the current lexical bindings.
pub(crate) fn apply(
    root: ArrayRef,
    expr: &BoundExpression,
    ctx: &mut ApplyCtx,
) -> VortexResult<ArrayRef> {
    let (scalar_fn, children) = match expr {
        BoundExpression::Root { dtype } => {
            vortex_ensure!(
                root.dtype() == dtype,
                "expression root dtype {dtype} does not match input array dtype {}",
                root.dtype()
            );
            return Ok(root);
        }
        BoundExpression::Variable {
            dtype,
            variable,
            variable_ref,
            ..
        } => {
            let array = ctx
                .binding(*variable_ref)
                .cloned()
                .ok_or_else(|| vortex_err!("cannot apply unbound variable '{variable}'"))?;
            vortex_ensure!(
                array.dtype() == dtype,
                "binding for variable '{variable}' has dtype {}, expected {dtype}",
                array.dtype()
            );
            vortex_ensure!(
                array.len() == root.len(),
                "binding for variable '{variable}' has length {}, expected {}",
                array.len(),
                root.len()
            );
            return Ok(array);
        }
        BoundExpression::Lambda { .. } => {
            return Err(vortex_err!(
                "a lambda can be applied only by the binder that established its scope"
            ));
        }
        BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } => (scalar_fn, children),
    };

    if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
        return Ok(ConstantArray::new(scalar.clone(), root.len()).into_array());
    }

    let children = children
        .iter()
        .map(|child| apply(root.clone(), child, ctx))
        .try_collect()?;
    let array =
        ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, root.len())?.into_array();
    array.optimize()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Scope;
    use crate::expr::checked_add;
    use crate::expr::lit;
    use crate::expr::var;

    #[test]
    fn applies_external_variable_binding() -> VortexResult<()> {
        let root = buffer![0_i32, 0, 0].into_array();
        let values = buffer![10_i32, 20, 30].into_array();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scope =
            Scope::new(dtype).with_bindings([(Variable::new("value"), values.dtype().clone())])?;
        let expr = checked_add(var("value"), lit(1_i32));

        let bound = expr.bind(&scope)?;
        let bindings = Bindings::from_scope(&scope, [(Variable::new("value"), values)])?;
        let result = root
            .apply_bound_with_bindings(&bound, &bindings)?
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?
            .into_array();

        assert_arrays_eq!(
            result,
            buffer![11_i32, 21, 31].into_array(),
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn bound_application_checks_external_binding_length() -> VortexResult<()> {
        let root = buffer![0_i32, 0, 0].into_array();
        let values = buffer![10_i32, 20].into_array();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scope =
            Scope::new(dtype).with_bindings([(Variable::new("value"), values.dtype().clone())])?;
        let bound = var("value").bind(&scope)?;
        let bindings = Bindings::from_scope(&scope, [(Variable::new("value"), values)])?;

        assert!(root.apply_bound_with_bindings(&bound, &bindings).is_err());
        Ok(())
    }

    #[test]
    fn bound_application_checks_root_dtype() -> VortexResult<()> {
        let expected = DType::Primitive(PType::I32, Nullability::NonNullable);
        let bound = crate::expr::root().bind(&expected)?;
        let root = buffer![0_i64, 0].into_array();

        assert!(root.apply_bound(&bound).is_err());
        Ok(())
    }

    #[test]
    fn external_bindings_check_scope_root_dtype() -> VortexResult<()> {
        let root = buffer![0_i64, 0].into_array();
        let values = buffer![10_i32, 20].into_array();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scope =
            Scope::new(dtype).with_bindings([(Variable::new("value"), values.dtype().clone())])?;
        let bound = var("value").bind(&scope)?;
        let bindings = Bindings::from_scope(&scope, [(Variable::new("value"), values)])?;

        assert!(root.apply_bound_with_bindings(&bound, &bindings).is_err());
        Ok(())
    }
}
