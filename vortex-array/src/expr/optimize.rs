// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::Scope;
use crate::expr::transform::match_between::find_between;
use crate::scalar_fn::ExpressionReduceNode;
use crate::scalar_fn::SimplifyCtx;

impl Expression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify_untyped` - type-independent simplifications
    /// 2. `simplify` - type-aware simplifications
    /// 3. `reduce` - abstract reduction rules via `ReduceNode`
    pub fn optimize(&self, scope: &DType) -> VortexResult<Expression> {
        self.optimize_scope(&Scope::new(scope.clone()))
    }

    /// Optimize the root expression node against a lexical scope.
    pub fn optimize_scope(&self, scope: &Scope) -> VortexResult<Expression> {
        let cache = SimplifyCache::new(scope);
        Ok(self.try_optimize(&cache)?.unwrap_or_else(|| self.clone()))
    }

    /// Apply this node's own untyped simplification rule, if it has one.
    ///
    /// Non-scalar nodes carry no rules, so they never simplify.
    fn simplify_untyped_node(&self) -> VortexResult<Option<Expression>> {
        match self {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.simplify_untyped(self),
            _ => Ok(None),
        }
    }

    /// Apply this node's own type-aware simplification rule, if it has one.
    fn simplify_node(&self, ctx: &dyn SimplifyCtx) -> VortexResult<Option<Expression>> {
        match self {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.simplify(self, ctx),
            _ => Ok(None),
        }
    }

    /// Apply this node's own abstract reduction rule, if it has one.
    fn reduce_node<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<Option<ExpressionReduceNode<'a>>> {
        match self {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.reduce_expression(node),
            _ => Ok(None),
        }
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    fn try_optimize(&self, cache: &SimplifyCache<'_>) -> VortexResult<Option<Expression>> {
        // Copy-on-write: `current` stays None until a rule fires, so unchanged nodes (the common
        // case) are never cloned.
        let mut current: Option<Expression> = None;
        let mut loop_counter = 0;

        loop {
            if loop_counter > 100 {
                vortex_error::vortex_bail!(
                    "Exceeded maximum optimization iterations (possible infinite loop)"
                );
            }
            loop_counter += 1;

            let expr = current.as_ref().unwrap_or(self);
            let mut changed = false;

            // Try simplify_untyped
            if let Some(simplified) = expr.simplify_untyped_node()? {
                current = Some(simplified);
                changed = true;
            }

            // Try simplify (typed)
            let expr = current.as_ref().unwrap_or(self);
            if let Some(simplified) = expr.simplify_node(cache)? {
                current = Some(simplified);
                changed = true;
            }

            // Try reduce via ReduceNode. The node borrows the expression and scope, so
            // constructing it is free; the block scopes the borrows so `current` can be updated.
            let reduced = {
                let expr = current.as_ref().unwrap_or(self);
                let reduce_node = ExpressionReduceNode::new(expr, cache.scope);
                expr.reduce_node(&reduce_node)?
                    .map(ExpressionReduceNode::into_expression)
            };
            if let Some(reduced_expr) = reduced {
                current = Some(reduced_expr);
                changed = true;
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    /// Optimize the entire expression tree recursively.
    ///
    /// Optimizes children first (bottom-up), then optimizes the root.
    pub fn optimize_recursive(&self, scope: &DType) -> VortexResult<Expression> {
        self.optimize_recursive_scope(&Scope::new(scope.clone()))
    }

    /// Optimize the entire expression tree recursively against a lexical scope.
    ///
    /// Variables are resolved from `scope` during type-aware simplification. Higher-order
    /// functions establish a child scope for each lambda body, so the body is optimized under its
    /// own root dtype and parameter frame while keeping enclosing captures available.
    pub fn optimize_recursive_scope(&self, scope: &Scope) -> VortexResult<Expression> {
        Ok(self
            .clone()
            .try_optimize_recursive_scope(scope)?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the entire expression tree recursively.
    pub fn try_optimize_recursive(&self, scope: &DType) -> VortexResult<Option<Expression>> {
        self.try_optimize_recursive_scope(&Scope::new(scope.clone()))
    }

    /// Try to optimize the entire expression tree recursively against a lexical scope.
    pub fn try_optimize_recursive_scope(&self, scope: &Scope) -> VortexResult<Option<Expression>> {
        let cache = SimplifyCache::new(scope);
        let result = self.try_optimize_recursive_inner(&cache)?;

        // Apply the between optimization once at the top level only.
        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        //  to CNF?
        Ok(Some(find_between(result.unwrap_or_else(|| self.clone()))))
    }

    fn try_optimize_recursive_inner(
        &self,
        cache: &SimplifyCache<'_>,
    ) -> VortexResult<Option<Expression>> {
        // First optimize the root
        let mut current = self.try_optimize(cache)?;

        // Then recursively optimize children. The new children vector is only allocated once a
        // child actually changes, so fully-optimized subtrees cost no allocations.
        let expr = current.as_ref().unwrap_or(self);
        let children = expr.children();
        let mut new_children: Option<Vec<Expression>> = None;
        for (idx, child) in children.iter().enumerate() {
            if let Some(optimized) = child.try_optimize_recursive_inner(cache)? {
                new_children
                    .get_or_insert_with(|| children[..idx].to_vec())
                    .push(optimized);
            } else if let Some(new_children) = new_children.as_mut() {
                new_children.push(child.clone());
            }
        }

        if let Some(new_children) = new_children {
            let updated = expr.clone().with_children(new_children)?;

            // After updating children, try to optimize root again
            current = Some(updated.try_optimize(cache)?.unwrap_or(updated));
        }

        Ok(current)
    }
}

struct SimplifyCache<'a> {
    scope: &'a Scope,
    dtype_cache: RefCell<HashMap<Expression, DType>>,
}

impl<'a> SimplifyCache<'a> {
    fn new(scope: &'a Scope) -> Self {
        Self {
            scope,
            dtype_cache: RefCell::new(HashMap::new()),
        }
    }
}

impl SimplifyCtx for SimplifyCache<'_> {
    fn return_dtype(&self, expr: &Expression) -> VortexResult<DType> {
        // If the expression is "root", return the scope dtype
        if expr.is_root() {
            return Ok(self.scope.root().clone());
        }

        if let Some(variable) = expr.as_variable() {
            return self
                .scope
                .resolve(variable)
                .map(|(dtype, _)| dtype.clone())
                .ok_or_else(|| vortex_err!("unbound variable '{variable}'"));
        }

        if let Some(dtype) = self.dtype_cache.borrow().get(expr) {
            return Ok(dtype.clone());
        }

        let dtype = match expr {
            Expression::Scalar { scalar_fn, .. } => {
                let input_dtypes: Vec<_> = expr
                    .children()
                    .iter()
                    .map(|child| self.return_dtype(child))
                    .try_collect()?;
                scalar_fn.return_dtype(&input_dtypes)?
            }
            Expression::Lambda(_) => {
                return Err(vortex_err!(
                    "a lambda has no standalone return dtype; it must be bound by a higher-order function"
                ));
            }
            Expression::Root | Expression::Variable(_) => unreachable!("handled above"),
        };
        self.dtype_cache
            .borrow_mut()
            .insert(expr.clone(), dtype.clone());

        Ok(dtype)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Scope;
    use crate::expr::Variable;
    use crate::expr::cast;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::lt_eq;
    use crate::expr::or;
    use crate::expr::root;
    use crate::expr::test_harness::struct_dtype;
    use crate::expr::var;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::literal::Literal;

    #[test]
    fn optimize_or_chain_correctness() -> VortexResult<()> {
        let expr = or(
            eq(get_item("x", root()), lit(1i32)),
            eq(get_item("x", root()), lit(2i32)),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let optimized = expr.optimize_recursive(&scope)?;

        let s = optimized.to_string();
        assert!(s.contains("$.x"), "expected $.x in {s}");
        assert!(s.contains("1i32") || s.contains('1'), "expected 1 in {s}");
        assert!(s.contains("2i32") || s.contains('2'), "expected 2 in {s}");
        Ok(())
    }

    #[test]
    fn optimize_folds_cast_of_literal_in_comparison() -> VortexResult<()> {
        let expr = lt_eq(
            get_item("x", root()),
            cast(
                lit(3i32),
                DType::Primitive(PType::F64, Nullability::NonNullable),
            ),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::F64, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let optimized = expr.optimize_recursive(&scope)?;

        // Prune rules pattern-match a bare Literal on the comparison RHS; a cast wrapper
        // silently disables pruning.
        let rhs = optimized
            .child(1)
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal RHS, got {optimized}"))?;
        assert_eq!(rhs, &Scalar::primitive(3.0f64, Nullability::NonNullable));
        Ok(())
    }

    #[test]
    fn optimization_folds_a_literal_cast() -> VortexResult<()> {
        let expr = lt_eq(
            col("a"),
            cast(
                lit(3_i32),
                DType::Primitive(PType::I64, Nullability::NonNullable),
            ),
        );
        let optimized = expr.optimize_recursive(&struct_dtype())?;
        assert_ne!(optimized, expr, "casting a literal should fold");
        Ok(())
    }

    #[test]
    fn optimization_resolves_variables_from_a_lexical_scope() -> VortexResult<()> {
        let scope = Scope::new(DType::Null).with_bindings([(
            Variable::new("value"),
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )])?;
        let expression = eq(var("value"), lit(42_i32));

        let optimized = expression.optimize_recursive_scope(&scope)?;
        assert_eq!(optimized, expression);
        Ok(())
    }
}
