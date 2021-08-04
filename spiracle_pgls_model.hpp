
// Code generated by stanc v2.26.0
#include <stan/model/model_header.hpp>
namespace spiracle_pgls_model_model_namespace {


inline void validate_positive_index(const char* var_name, const char* expr,
                                    int val) {
  if (val < 1) {
    std::stringstream msg;
    msg << "Found dimension size less than one in simplex declaration"
        << "; variable=" << var_name << "; dimension size expression=" << expr
        << "; expression value=" << val;
    std::string msg_str(msg.str());
    throw std::invalid_argument(msg_str.c_str());
  }
}

inline void validate_unit_vector_index(const char* var_name, const char* expr,
                                       int val) {
  if (val <= 1) {
    std::stringstream msg;
    if (val == 1) {
      msg << "Found dimension size one in unit vector declaration."
          << " One-dimensional unit vector is discrete"
          << " but the target distribution must be continuous."
          << " variable=" << var_name << "; dimension size expression=" << expr;
    } else {
      msg << "Found dimension size less than one in unit vector declaration"
          << "; variable=" << var_name << "; dimension size expression=" << expr
          << "; expression value=" << val;
    }
    std::string msg_str(msg.str());
    throw std::invalid_argument(msg_str.c_str());
  }
}


using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using std::pow;
using stan::io::dump;
using stan::math::lgamma;
using stan::model::model_base_crtp;
using stan::model::rvalue;
using stan::model::cons_list;
using stan::model::index_uni;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::nil_index_list;
using namespace stan::math;
using stan::math::pow; 

stan::math::profile_map profiles__;
static int current_statement__= 0;
static const std::vector<string> locations_array__ = {" (found before start of program)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 17, column 2 to column 9)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 18, column 2 to column 9)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 19, column 2 to column 32)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 20, column 2 to column 22)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 24, column 2 to column 15)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 25, column 2 to column 19)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 28, column 4 to column 26)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 27, column 17 to line 29, column 3)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 27, column 2 to line 29, column 3)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 36, column 8 to column 41)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 34, column 8 to column 42)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 33, column 6 to line 36, column 41)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 32, column 19 to line 37, column 5)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 32, column 4 to line 37, column 5)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 31, column 17 to line 38, column 3)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 31, column 2 to line 38, column 3)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 54, column 2 to column 18)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 55, column 2 to column 19)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 58, column 4 to column 30)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 57, column 17 to line 59, column 3)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 57, column 2 to line 59, column 3)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 61, column 2 to column 40)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 43, column 2 to column 26)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 44, column 2 to column 24)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 47, column 2 to column 27)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 48, column 2 to column 26)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 49, column 2 to column 28)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 9, column 2 to column 17)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 10, column 9 to column 10)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 10, column 2 to column 14)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 11, column 9 to column 10)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 11, column 2 to column 14)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 12, column 2 to column 14)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 13, column 9 to column 10)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 13, column 12 to column 13)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 13, column 2 to column 25)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 24, column 9 to column 10)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 25, column 9 to column 10)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 25, column 12 to column 13)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 54, column 9 to column 10)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 55, column 9 to column 10)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 3, column 4 to column 25)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 4, column 4 to column 15)",
                                                      " (in 'C:/Users/jwagne/git/spiracle_scaling/spiracle_pgls_model.stan', line 2, column 35 to line 5, column 3)"};


template <typename T0__, typename T1__, typename T2__>
stan::promote_args_t<T0__, T1__,
T2__>
f(const T0__& x_i, const T1__& a, const T2__& b, std::ostream* pstream__) {
  using local_scalar_t__ = stan::promote_args_t<T0__, T1__, T2__>;
  const static bool propto__ = true;
  (void) propto__;
  local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
  (void) DUMMY_VAR__;  // suppress unused var warning
  
  try {
    local_scalar_t__ y_i;
    y_i = DUMMY_VAR__;
    
    current_statement__ = 42;
    y_i = ((x_i * a) + b);
    current_statement__ = 43;
    return y_i;
  } catch (const std::exception& e) {
    stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
  }
  
}

struct f_functor__ {
template <typename T0__, typename T1__, typename T2__>
stan::promote_args_t<T0__, T1__,
T2__>
operator()(const T0__& x_i, const T1__& a, const T2__& b,
           std::ostream* pstream__)  const 
{
return f(x_i, a, b, pstream__);
}
};

class spiracle_pgls_model_model final : public model_base_crtp<spiracle_pgls_model_model> {

 private:
  int N;
  Eigen::Matrix<double, -1, 1> x;
  Eigen::Matrix<double, -1, 1> y;
  double priora;
  Eigen::Matrix<double, -1, -1> cov_phylo;
 
 public:
  ~spiracle_pgls_model_model() { }
  
  inline std::string model_name() const final { return "spiracle_pgls_model_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.26.0", "stancflags = "};
  }
  
  
  spiracle_pgls_model_model(stan::io::var_context& context__,
                            unsigned int random_seed__ = 0,
                            std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static const char* function__ = "spiracle_pgls_model_model_namespace::spiracle_pgls_model_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      current_statement__ = 28;
      context__.validate_dims("data initialization","N","int",
          context__.to_vec());
      N = std::numeric_limits<int>::min();
      
      current_statement__ = 28;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 28;
      current_statement__ = 28;
      check_greater_or_equal(function__, "N", N, 1);
      current_statement__ = 29;
      validate_non_negative_index("x", "N", N);
      current_statement__ = 30;
      context__.validate_dims("data initialization","x","double",
          context__.to_vec(N));
      x = Eigen::Matrix<double, -1, 1>(N);
      stan::math::fill(x, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> x_flat__;
        current_statement__ = 30;
        assign(x_flat__, nil_index_list(), context__.vals_r("x"),
          "assigning variable x_flat__");
        current_statement__ = 30;
        pos__ = 1;
        current_statement__ = 30;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 30;
          assign(x, cons_list(index_uni(sym1__), nil_index_list()),
            x_flat__[(pos__ - 1)], "assigning variable x");
          current_statement__ = 30;
          pos__ = (pos__ + 1);}
      }
      current_statement__ = 31;
      validate_non_negative_index("y", "N", N);
      current_statement__ = 32;
      context__.validate_dims("data initialization","y","double",
          context__.to_vec(N));
      y = Eigen::Matrix<double, -1, 1>(N);
      stan::math::fill(y, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> y_flat__;
        current_statement__ = 32;
        assign(y_flat__, nil_index_list(), context__.vals_r("y"),
          "assigning variable y_flat__");
        current_statement__ = 32;
        pos__ = 1;
        current_statement__ = 32;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 32;
          assign(y, cons_list(index_uni(sym1__), nil_index_list()),
            y_flat__[(pos__ - 1)], "assigning variable y");
          current_statement__ = 32;
          pos__ = (pos__ + 1);}
      }
      current_statement__ = 33;
      context__.validate_dims("data initialization","priora","double",
          context__.to_vec());
      priora = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 33;
      priora = context__.vals_r("priora")[(1 - 1)];
      current_statement__ = 34;
      validate_non_negative_index("cov_phylo", "N", N);
      current_statement__ = 35;
      validate_non_negative_index("cov_phylo", "N", N);
      current_statement__ = 36;
      context__.validate_dims("data initialization","cov_phylo","double",
          context__.to_vec(N, N));
      cov_phylo = Eigen::Matrix<double, -1, -1>(N, N);
      stan::math::fill(cov_phylo, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> cov_phylo_flat__;
        current_statement__ = 36;
        assign(cov_phylo_flat__, nil_index_list(),
          context__.vals_r("cov_phylo"),
          "assigning variable cov_phylo_flat__");
        current_statement__ = 36;
        pos__ = 1;
        current_statement__ = 36;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 36;
          for (int sym2__ = 1; sym2__ <= N; ++sym2__) {
            current_statement__ = 36;
            assign(cov_phylo,
              cons_list(index_uni(sym2__),
                cons_list(index_uni(sym1__), nil_index_list())),
              cov_phylo_flat__[(pos__ - 1)], "assigning variable cov_phylo");
            current_statement__ = 36;
            pos__ = (pos__ + 1);}}
      }
      current_statement__ = 37;
      validate_non_negative_index("mu", "N", N);
      current_statement__ = 38;
      validate_non_negative_index("cov", "N", N);
      current_statement__ = 39;
      validate_non_negative_index("cov", "N", N);
      current_statement__ = 40;
      validate_non_negative_index("y_ppc", "N", N);
      current_statement__ = 41;
      validate_non_negative_index("mu_ppc", "N", N);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    num_params_r__ = 0U;
    
    try {
      num_params_r__ += 1;
      num_params_r__ += 1;
      num_params_r__ += 1;
      num_params_r__ += 1;
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
  }
  template <bool propto__, bool jacobian__, typename VecR, typename VecI, stan::require_vector_like_t<VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    static const char* function__ = "spiracle_pgls_model_model_namespace::log_prob";
(void) function__;  // suppress unused var warning

    stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning

    
    try {
      local_scalar_t__ a;
      a = DUMMY_VAR__;
      
      current_statement__ = 1;
      a = in__.scalar();
      local_scalar_t__ b;
      b = DUMMY_VAR__;
      
      current_statement__ = 2;
      b = in__.scalar();
      local_scalar_t__ lambda;
      lambda = DUMMY_VAR__;
      
      current_statement__ = 3;
      lambda = in__.scalar();
      current_statement__ = 3;
      if (jacobian__) {
        current_statement__ = 3;
        lambda = stan::math::lub_constrain(lambda, 0, 1, lp__);
      } else {
        current_statement__ = 3;
        lambda = stan::math::lub_constrain(lambda, 0, 1);
      }
      local_scalar_t__ sigma;
      sigma = DUMMY_VAR__;
      
      current_statement__ = 4;
      sigma = in__.scalar();
      current_statement__ = 4;
      if (jacobian__) {
        current_statement__ = 4;
        sigma = stan::math::lb_constrain(sigma, 0, lp__);
      } else {
        current_statement__ = 4;
        sigma = stan::math::lb_constrain(sigma, 0);
      }
      Eigen::Matrix<local_scalar_t__, -1, 1> mu;
      mu = Eigen::Matrix<local_scalar_t__, -1, 1>(N);
      stan::math::fill(mu, DUMMY_VAR__);
      
      Eigen::Matrix<local_scalar_t__, -1, -1> cov;
      cov = Eigen::Matrix<local_scalar_t__, -1, -1>(N, N);
      stan::math::fill(cov, DUMMY_VAR__);
      
      current_statement__ = 9;
      for (int i = 1; i <= N; ++i) {
        current_statement__ = 7;
        assign(mu, cons_list(index_uni(i), nil_index_list()),
          f(x[(i - 1)], a, b, pstream__), "assigning variable mu");}
      current_statement__ = 16;
      for (int i = 1; i <= N; ++i) {
        current_statement__ = 14;
        for (int j = 1; j <= N; ++j) {
          current_statement__ = 12;
          if (logical_eq(logical_negation(i), j)) {
            current_statement__ = 11;
            assign(cov,
              cons_list(index_uni(i),
                cons_list(index_uni(j), nil_index_list())),
              (rvalue(cov_phylo,
                 cons_list(index_uni(i),
                   cons_list(index_uni(j), nil_index_list())), "cov_phylo") *
                lambda), "assigning variable cov");
          } else {
            current_statement__ = 10;
            assign(cov,
              cons_list(index_uni(i),
                cons_list(index_uni(j), nil_index_list())),
              (rvalue(cov_phylo,
                 cons_list(index_uni(i),
                   cons_list(index_uni(j), nil_index_list())), "cov_phylo") *
                sigma), "assigning variable cov");
          }}}
      {
        current_statement__ = 23;
        lp_accum__.add(normal_lpdf<propto__>(a, priora, 0.3));
        current_statement__ = 24;
        lp_accum__.add(normal_lpdf<propto__>(b, -1.0, 2.0));
        current_statement__ = 25;
        lp_accum__.add(normal_lpdf<propto__>(sigma, 0.0, 1.0));
        current_statement__ = 26;
        lp_accum__.add(beta_lpdf<propto__>(lambda, 1.4, 1.4));
        current_statement__ = 27;
        lp_accum__.add(multi_normal_lpdf<propto__>(y, mu, cov));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, stan::require_std_vector_vt<std::is_floating_point, VecVar>* = nullptr>
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.resize(0);
    stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
    static const char* function__ = "spiracle_pgls_model_model_namespace::write_array";
(void) function__;  // suppress unused var warning

    (void) function__;  // suppress unused var warning

    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning

    
    try {
      double a;
      a = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 1;
      a = in__.scalar();
      double b;
      b = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 2;
      b = in__.scalar();
      double lambda;
      lambda = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 3;
      lambda = in__.scalar();
      current_statement__ = 3;
      lambda = stan::math::lub_constrain(lambda, 0, 1);
      double sigma;
      sigma = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 4;
      sigma = in__.scalar();
      current_statement__ = 4;
      sigma = stan::math::lb_constrain(sigma, 0);
      Eigen::Matrix<double, -1, 1> mu;
      mu = Eigen::Matrix<double, -1, 1>(N);
      stan::math::fill(mu, std::numeric_limits<double>::quiet_NaN());
      
      Eigen::Matrix<double, -1, -1> cov;
      cov = Eigen::Matrix<double, -1, -1>(N, N);
      stan::math::fill(cov, std::numeric_limits<double>::quiet_NaN());
      
      vars__.emplace_back(a);
      vars__.emplace_back(b);
      vars__.emplace_back(lambda);
      vars__.emplace_back(sigma);
      if (logical_negation((primitive_value(emit_transformed_parameters__) ||
            primitive_value(emit_generated_quantities__)))) {
        return ;
      } 
      current_statement__ = 9;
      for (int i = 1; i <= N; ++i) {
        current_statement__ = 7;
        assign(mu, cons_list(index_uni(i), nil_index_list()),
          f(x[(i - 1)], a, b, pstream__), "assigning variable mu");}
      current_statement__ = 16;
      for (int i = 1; i <= N; ++i) {
        current_statement__ = 14;
        for (int j = 1; j <= N; ++j) {
          current_statement__ = 12;
          if (logical_eq(logical_negation(i), j)) {
            current_statement__ = 11;
            assign(cov,
              cons_list(index_uni(i),
                cons_list(index_uni(j), nil_index_list())),
              (rvalue(cov_phylo,
                 cons_list(index_uni(i),
                   cons_list(index_uni(j), nil_index_list())), "cov_phylo") *
                lambda), "assigning variable cov");
          } else {
            current_statement__ = 10;
            assign(cov,
              cons_list(index_uni(i),
                cons_list(index_uni(j), nil_index_list())),
              (rvalue(cov_phylo,
                 cons_list(index_uni(i),
                   cons_list(index_uni(j), nil_index_list())), "cov_phylo") *
                sigma), "assigning variable cov");
          }}}
      if (emit_transformed_parameters__) {
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          vars__.emplace_back(mu[(sym1__ - 1)]);}
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          for (int sym2__ = 1; sym2__ <= N; ++sym2__) {
            vars__.emplace_back(
              rvalue(cov,
                cons_list(index_uni(sym2__),
                  cons_list(index_uni(sym1__), nil_index_list())), "cov"));}}
      } 
      if (logical_negation(emit_generated_quantities__)) {
        return ;
      } 
      Eigen::Matrix<double, -1, 1> y_ppc;
      y_ppc = Eigen::Matrix<double, -1, 1>(N);
      stan::math::fill(y_ppc, std::numeric_limits<double>::quiet_NaN());
      
      Eigen::Matrix<double, -1, 1> mu_ppc;
      mu_ppc = Eigen::Matrix<double, -1, 1>(N);
      stan::math::fill(mu_ppc, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 21;
      for (int i = 1; i <= N; ++i) {
        current_statement__ = 19;
        assign(mu_ppc, cons_list(index_uni(i), nil_index_list()),
          f(x[(i - 1)], a, b, pstream__), "assigning variable mu_ppc");}
      current_statement__ = 22;
      assign(y_ppc, nil_index_list(),
        multi_normal_rng(mu_ppc, cov, base_rng__), "assigning variable y_ppc");
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(y_ppc[(sym1__ - 1)]);}
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(mu_ppc[(sym1__ - 1)]);}
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, stan::require_std_vector_t<VecVar>* = nullptr, stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void transform_inits_impl(const stan::io::var_context& context__,
                                   VecI& params_i__, VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.clear();
    vars__.reserve(num_params_r__);
    
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      double a;
      a = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 1;
      a = context__.vals_r("a")[(1 - 1)];
      double b;
      b = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 2;
      b = context__.vals_r("b")[(1 - 1)];
      double lambda;
      lambda = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 3;
      lambda = context__.vals_r("lambda")[(1 - 1)];
      double lambda_free__;
      lambda_free__ = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 3;
      lambda_free__ = stan::math::lub_free(lambda, 0, 1);
      double sigma;
      sigma = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 4;
      sigma = context__.vals_r("sigma")[(1 - 1)];
      double sigma_free__;
      sigma_free__ = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 4;
      sigma_free__ = stan::math::lb_free(sigma, 0);
      vars__.emplace_back(a);
      vars__.emplace_back(b);
      vars__.emplace_back(lambda_free__);
      vars__.emplace_back(sigma_free__);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__.clear();
    names__.emplace_back("a");
    names__.emplace_back("b");
    names__.emplace_back("lambda");
    names__.emplace_back("sigma");
    names__.emplace_back("mu");
    names__.emplace_back("cov");
    names__.emplace_back("y_ppc");
    names__.emplace_back("mu_ppc");
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    dimss__.clear();
    dimss__.emplace_back(std::vector<size_t>{});
    
    dimss__.emplace_back(std::vector<size_t>{});
    
    dimss__.emplace_back(std::vector<size_t>{});
    
    dimss__.emplace_back(std::vector<size_t>{});
    
    dimss__.emplace_back(std::vector<size_t>{static_cast<size_t>(N)});
    
    dimss__.emplace_back(std::vector<size_t>{static_cast<size_t>(N),
                                             static_cast<size_t>(N)});
    
    dimss__.emplace_back(std::vector<size_t>{static_cast<size_t>(N)});
    
    dimss__.emplace_back(std::vector<size_t>{static_cast<size_t>(N)});
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "a");
    param_names__.emplace_back(std::string() + "b");
    param_names__.emplace_back(std::string() + "lambda");
    param_names__.emplace_back(std::string() + "sigma");
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "mu" + '.' + std::to_string(sym1__));
        }}
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          for (int sym2__ = 1; sym2__ <= N; ++sym2__) {
            {
              param_names__.emplace_back(std::string() + "cov" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
            }}
        }}
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "y_ppc" + '.' + std::to_string(sym1__));
        }}
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "mu_ppc" + '.' + std::to_string(sym1__));
        }}
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "a");
    param_names__.emplace_back(std::string() + "b");
    param_names__.emplace_back(std::string() + "lambda");
    param_names__.emplace_back(std::string() + "sigma");
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "mu" + '.' + std::to_string(sym1__));
        }}
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          for (int sym2__ = 1; sym2__ <= N; ++sym2__) {
            {
              param_names__.emplace_back(std::string() + "cov" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
            }}
        }}
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "y_ppc" + '.' + std::to_string(sym1__));
        }}
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "mu_ppc" + '.' + std::to_string(sym1__));
        }}
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    stringstream s__;
    s__ << "[{\"name\":\"a\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"b\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"lambda\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"mu\",\"type\":{\"name\":\"vector\",\"length\":" << N << "},\"block\":\"transformed_parameters\"},{\"name\":\"cov\",\"type\":{\"name\":\"matrix\",\"rows\":" << N << ",\"cols\":" << N << "},\"block\":\"transformed_parameters\"},{\"name\":\"y_ppc\",\"type\":{\"name\":\"vector\",\"length\":" << N << "},\"block\":\"generated_quantities\"},{\"name\":\"mu_ppc\",\"type\":{\"name\":\"vector\",\"length\":" << N << "},\"block\":\"generated_quantities\"}]";
    return s__.str();
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    stringstream s__;
    s__ << "[{\"name\":\"a\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"b\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"lambda\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"mu\",\"type\":{\"name\":\"vector\",\"length\":" << N << "},\"block\":\"transformed_parameters\"},{\"name\":\"cov\",\"type\":{\"name\":\"matrix\",\"rows\":" << N << ",\"cols\":" << N << "},\"block\":\"transformed_parameters\"},{\"name\":\"y_ppc\",\"type\":{\"name\":\"vector\",\"length\":" << N << "},\"block\":\"generated_quantities\"},{\"name\":\"mu_ppc\",\"type\":{\"name\":\"vector\",\"length\":" << N << "},\"block\":\"generated_quantities\"}]";
    return s__.str();
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      std::vector<double> vars_vec(vars.size());
      std::vector<int> params_i;
      write_array_impl(base_rng, params_r, params_i, vars_vec,
          emit_transformed_parameters, emit_generated_quantities, pstream);
      vars.resize(vars_vec.size());
      for (int i = 0; i < vars.size(); ++i) {
        vars.coeffRef(i) = vars_vec[i];
      }
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      write_array_impl(base_rng, params_r, params_i, vars, emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }
  

    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits_impl(context, params_i, params_r_vec, pstream);
      params_r.resize(params_r_vec.size());
      for (int i = 0; i < params_r.size(); ++i) {
        params_r.coeffRef(i) = params_r_vec[i];
      }
    }
    inline void transform_inits(const stan::io::var_context& context,
                                std::vector<int>& params_i,
                                std::vector<double>& vars,
                                std::ostream* pstream = nullptr) const final {
      transform_inits_impl(context, params_i, vars, pstream);
    }        

};
}

using stan_model = spiracle_pgls_model_model_namespace::spiracle_pgls_model_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return spiracle_pgls_model_model_namespace::profiles__;
}

#endif


