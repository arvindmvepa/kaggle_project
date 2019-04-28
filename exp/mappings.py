from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, Lars, LassoLars, \
    OrthogonalMatchingPursuit, SGDRegressor, PassiveAggressiveRegressor, TheilSenRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


# algorithm mappings
alg_map = {
    "lr": (LinearRegression, "sklearn"),
    "ridge": (Ridge, "sklearn"),
    "lasso": (Lasso, "sklearn"),
    "mtlasso": (MultiTaskLasso, "sklearn"),
    "elastic": (ElasticNet, "sklearn"),
    "lars": (Lars, "sklearn"),
    "llars": (LassoLars, "sklearn"),
    "omp": (OrthogonalMatchingPursuit, "sklearn"),
    "sgdreg": (SGDRegressor, "sklearn"),
    "pareg": (PassiveAggressiveRegressor, "sklearn"),
    "tsreg": (TheilSenRegressor, "sklearn"),
    "hreg": (HuberRegressor, "sklearn"),
    "kreg": (KernelRidge, "sklearn"),
    "svr": (SVR, "sklearn"),
    "nsvr": (NuSVR, "sklearn"),
    "lsvr": (LinearSVR, "sklearn"),
    "knreg": (KNeighborsRegressor, "sklearn"),
    "rnreg": (RadiusNeighborsRegressor, "sklearn"),
    "gpreg": (GaussianProcessRegressor, "sklearn"),
    "plsreg": (PLSRegression, "sklearn"),
    "dtreg": (DecisionTreeRegressor, "sklearn"),
    "bagreg": (BaggingRegressor, "sklearn"),
    "rfreg": (RandomForestRegressor, "sklearn"),
    "etreg": (ExtraTreesRegressor, "sklearn"),
    "abreg": (AdaBoostRegressor, "sklearn"),
    "gbreg": (GradientBoostingRegressor, "sklearn"),
    "mlpreg": (MLPRegressor, "sklearn"),
    "lgb": (lgb.LGBMRegressor, "sklearn"),
    "xgb": (xgb, "xgb"),
    "cat": (CatBoostRegressor, "cat"),
}
