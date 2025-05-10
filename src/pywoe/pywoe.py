import numpy
import scipy.stats
import lightgbm
import sklearn.base
import multiprocess as mp
import pandas as pd
import functools
import warnings
import typing

class PyWOE(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    A class to perform weight of evidence (WoE) binning on a dataset.

    This class provides functionality to bin numerical features into categorical
    bins based on their WoE values, which are calculated as the log of the ratio
    of the bads to the goods in each bin.

    Categorical features are supported.

    Standard sklearn pipeline compatible. (I.e. implements fit and transform methods.)
    """
    def __init__(
            self,
            n_jobs:int=-1,
            min_leave_freq:float=0.025,
            random_state:typing.Union[int, None]=None,
        ) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self._splits = None
        self._names = None
        self._woe_df = None
        self.min_leave_freq = min_leave_freq
        self.random_state = random_state

    def fit(self, X:typing.Union[pd.DataFrame, pd.Series, numpy.ndarray], y:typing.Union[pd.Series, numpy.ndarray]) -> "PyWOE":
        self._generate_splits(X, y)
        return self
    
    def transform(self, X:typing.Union[pd.DataFrame, pd.Series, numpy.ndarray]) -> typing.Union[pd.DataFrame, pd.Series, numpy.ndarray]:
        def _transform_categorical(x: pd.DataFrame):
            missing_categories = [cat not in self._splits[x.columns[0]]["cuts"] for cat in x[x.columns[0]].unique()]
            if any(missing_categories):
                raise ValueError(f"Column {x.columns[0]} has categories ({', '.join(x[x.columns[0]].unique()[missing_categories])}) not present in the splits.")
            x_transformed = x.merge(self._woe_df.query(f"variable == '{x.columns[0]}'")[["group", "woe"]], left_on=x.columns[0], right_on="group", how="left")
            x_transformed.drop(columns=[x.columns[0], "group"], inplace=True)
            x_transformed.rename(columns={"woe": x.columns[0]}, inplace=True)
            return x_transformed
        def _transform_numerical(x: pd.DataFrame):
            x_transformed = x.copy()
            x_transformed[x.columns[0]] = pd.cut(x[x.columns[0]], bins=self._splits[x.columns[0]]["cuts"], labels=self._splits[x.columns[0]]["labels"]).fillna(self._splits[x.columns[0]]["na_label"])
            x_transformed = x_transformed.merge(self._woe_df.query(f"variable == '{x.columns[0]}'")[["group", "woe"]], left_on=x.columns[0], right_on="group", how="left")
            x_transformed.drop(columns=[x.columns[0], "group"], inplace=True)
            x_transformed.rename(columns={"woe": x.columns[0]}, inplace=True)
            return x_transformed
        if self._splits is None or self._woe_df is None:
            raise ValueError("Model not fitted.")
        X_pandas = self._cast_to_pandas(X)
        missing_columns = [col not in self._splits for col in X_pandas.columns]
        if any(missing_columns):
            raise ValueError(f"Column {', '.join([X_pandas.columns[i] for i in range(len(X_pandas.columns)) if missing_columns[i]])} not present in the splits.")
        with mp.Pool(self.n_jobs if self.n_jobs > 0 else mp.cpu_count()) as pool:
            X_transformed = pd.concat(pool.map(
                lambda x: _transform_categorical(X_pandas[[x]]) if X_pandas[x].dtype in ["object", "category", "string"] else _transform_numerical(X_pandas[[x]]), X_pandas.columns
            ), axis=1)
        # X_transformed = pd.concat([
        #     *map(
        #         lambda x: _transform_categorical(X_pandas[[x]]) if X_pandas[x].dtype in ["object", "category", "string"] else _transform_numerical(X_pandas[[x]]), 
        #         X_pandas.columns
        #     )
        # ], axis=1)
        return X_transformed
    
    def fit_transform(
            self,
            X:typing.Union[pd.DataFrame, pd.Series, numpy.ndarray],
            y:typing.Union[pd.Series, numpy.ndarray]
        ) -> typing.Union[pd.DataFrame, pd.Series, numpy.ndarray]:
        self.fit(X, y)
        return self.transform(X)
    
    def woe_summary(self) -> pd.DataFrame:
        return self._woe_df
    
    def splits(self) -> typing.Dict[str, typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]]:
        return self._splits
    
    def _cast_to_pandas(self, X:typing.Union[pd.DataFrame, pd.Series, numpy.ndarray]) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X_pandas = pd.DataFrame(X)
            X_pandas.columns = self._get_names(X)
            return X_pandas
        else:
            X.columns = self._get_names(X)
            return X
    
    def _get_names(self, X:typing.Union[pd.DataFrame, pd.Series, numpy.ndarray]) -> typing.List[str]:
        if self._names is None:
            if hasattr(X, "name"):
                self._names = [str(X.name), ]
            elif hasattr(X, "columns"):
                self._names = [str(col) for col in X.columns.tolist()]
            elif isinstance(X, numpy.ndarray):
                if X.ndim == 1:
                    self._names = ["0", ]
                elif X.ndim == 2:
                    self._names = [str(i) for i in range(X.shape[1])]
                else:
                    raise ValueError(f"Invalid input numpy.ndarray shape: {X.shape}")
            else:
                raise ValueError(f"Invalid input type: {type(X)}")
        return self._names
    
    def _generate_splits(
            self,
            X:typing.Union[pd.DataFrame, pd.Series, numpy.ndarray],
            y:typing.Union[pd.Series, numpy.ndarray]
        ) -> None:
        def _get_splits(x: pd.DataFrame):
            if x.dtypes.tolist()[0] in ["object", "category", "string"]:
                stats, cuts = _get_splits_categorical(x)
            else:
                stats, cuts = _get_lgbm_splits(x)
            stats["non_events"] = stats["freq"] - stats["events"] + 0.5
            stats["events"] = stats["events"] + 0.5
            stats["eventrate"] = stats["events"] / stats["freq"]
            stats["woe"] = numpy.log(
                (stats["events"] / stats["events"].sum())
                / (stats["non_events"] / stats["non_events"].sum())
            )
            stats["iv"] = (
                (stats["events"] / stats["events"].sum()) - 
                (stats["non_events"] / stats["non_events"].sum())
            ) * stats["woe"]
            stats["group"] = cuts["labels"]
            # print(cuts)
            return stats, cuts

        def _get_lgbm_splits(x: pd.DataFrame):
            cor = scipy.stats.spearmanr(x, y)[0]
            con = "1" if cor > 0 else "-1"
            con = "1"

            gbm = lightgbm.LGBMRegressor(
                num_leaves=100,
                min_child_samples=int(numpy.ceil(len(x) * self.min_leave_freq)),
                n_estimators=1,
                random_state=self.random_state,
                monotone_constraints=con,
                verbose=-1,
                n_jobs=1,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                gbm.fit(x, y)

            predictions = pd.DataFrame({
                "variable": str(x.columns[0]),
                "prediction": gbm.predict(x),
                "x": x[x.columns[0]],
                "y": y,
            })
            predictions = predictions.sort_values(by=["variable", "prediction"])
            prediction_stats = predictions.groupby(["variable", "prediction"], observed=True).agg(
                freq=("y", "count"),
                events=("y", "sum"),
                minx=("x", "min"),
                maxx=("x", "max"),
                nas=("x", lambda x: x.isna().sum()),
            )
            cuts = self._generate_cuts(
                prediction_stats["minx"],
                prediction_stats["nas"],
            )
            return (prediction_stats.reset_index().drop(columns=["prediction", "minx", "maxx"]), cuts)
            # return (prediction_stats.reset_index(), cuts)

        def _get_splits_categorical(x: pd.DataFrame):
            x = x.copy()
            x["y"] = y
            x[x.columns[0]] = x[x.columns[0]].astype("category")
            if x[x.columns[0]].isna().sum() > 0:
                x[x.columns[0]] = x[x.columns[0]].cat.add_categories("NA")
                x.loc[x[x.columns[0]].isna(), x.columns[0]] = "NA"
            x["variable"] = str(x.columns[0])
            stats = x.groupby(["variable", x.columns[0]], observed=True).agg(
                freq=("y", "count"),
                events=("y", "sum"),
            )
            categories = x[x.columns[0]].cat.categories.tolist()
            categories.sort()
            cuts = {
                "cuts": categories,
                "na_label": "NA",
                "labels": categories,
            }
            return (stats.reset_index().sort_values(by=x.columns[0]).drop(columns=[x.columns[0]]), cuts)

        X_pandas = self._cast_to_pandas(X)
        # with mp.Pool(self.n_jobs if self.n_jobs > 0 else mp.cpu_count()) as pool:
        #     splits = pool.map(lambda column: _get_splits(X_pandas[[column]]), X_pandas.columns)
        splits = [*map(lambda column: _get_splits(X_pandas[[column]]), X_pandas.columns.tolist())]
        self._woe_df = pd.concat([woe_df for woe_df, _ in splits], axis=0).reset_index(drop=True)
        self._splits = {col: cuts for col, (_, cuts) in zip(X_pandas.columns, splits)}


    
    @staticmethod
    def _generate_cuts(minx: pd.Series, nas: pd.Series) -> typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]:
        minx_nonna = minx[~minx.isna()]
        if len(minx_nonna) < 1:
            cuts = [-numpy.inf, numpy.inf]
        else:
            cuts = [-numpy.inf, *minx_nonna[1:], numpy.inf]
        labels = [f"[{cuts[i]}, {cuts[i+1]})" for i in range(len(cuts)-1)]
        if nas.sum() > 0:
            na_index = numpy.where(nas > 0)[0][0]
            if numpy.isnan(minx.iloc[na_index]):
                na_label = "NA"
                labels.insert(na_index, na_label)
            else:
                labels[na_index] = f"{labels[na_index]} OR NA"
                na_label = labels[na_index]
        else:
            na_label = None
        return {
            "cuts": cuts,
            "na_label": na_label,
            "labels": labels,
        }
