# Adapted from FindMinimaxPolynomial documentation.

# We'll be using BigFloat, so increase it's precision somewhat
setprecision(BigFloat, 64*10)

import FindMinimaxPolynomial, Tulip, MathOptInterface

# Shorthands
const FMP = FindMinimaxPolynomial
const MMX = FMP.Minimax
const PPTI = FMP.PolynomialPassingThroughIntervals
const NE = FMP.NumericalErrorTypes
const to_poly = FMP.ToSparsePolynomial.to_sparse_polynomial
const mmx = MMX.minimax_polynomial
const MOI = MathOptInterface

function tulip_make_lp()
  lp = Tulip.Optimizer{BigFloat}()

  # Increase the Tulip iteration limit, just in case. The default limit is not good for
  # `BigFloat` problems.
  MOI.set(lp, MOI.RawOptimizerAttribute("IPM_IterationsLimit"), 1000)

  # Disable presolve
  MOI.set(lp, MOI.RawOptimizerAttribute("Presolve_Level"), 0)

  lp
end

function lanczos1(x)
    x == big"0.0" ? big"1.0" : (sin(pi * x) * sin(pi * x)) / (pi * pi * x * x)
end

# The interval on which we'll approximate
itv_lanczos = (big"0.0", big"1.0");

# Approximate the sine on the above interval with such a polynomial
opt = MMX.minimax_options(exact_points = (big"1.0",));
res_lanczos = mmx(tulip_make_lp, lanczos1, [itv_lanczos], 0:2:8, opt);

# Convert the coefficients into a polynomial for evaluation, plotting, etc.
poly_lanczos = to_poly(res_lanczos.mmx.coefs, 0:2:8);
println(poly_lanczos);

# Check that `mmx` converged to the optimal solution
==(map(Float64, res_lanczos.max_error)...) ||
  println("suboptimal solution for lanczos1, report a bug")


# Let's plot our approximation errors!

# using Gadfly
#
# err_sin(x) =
#   let x = BigFloat(x)
#     abs(1 - poly_sin(x)/sin(x))
#   end
#
# err_sin_max_const = Float64(last(res_sin.max_error))
#
# err_sin_max(x) = err_sin_max_const
#
# plot([err_sin, err_sin_max], Float64.(itv_sin)...)
#
# err_cos(x) =
#   let x = BigFloat(x)
#     abs(1 - poly_cos(x)/cos(x))
#   end
#
# err_cos_max_const = Float64(last(res_cos.max_error))
#
# err_cos_max(x) = err_cos_max_const
#
# plot([err_cos, err_cos_max], Float64.(itv_cos)...)
