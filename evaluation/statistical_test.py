from scipy import stats
from ern.utils import get_eval_results, get_test_results


def t_test(df, alpha_value=0.01, baseline_name="BASELINE"):
    significant_models = []
    df_baseline = df[df.ModelType == "BASELINE"].sort_values("ParkId")
    for model, df_model in df.groupby("ModelType"):
        if model == baseline_name:
            continue

        df_model = df_model.sort_values("ParkId")

        diff = df_baseline.RMSE - df_model.RMSE

        _, p = stats.shapiro(diff)
        if p > alpha_value:
            print(model)
            _, p = stats.ttest_rel(df_baseline.RMSE, df_model.RMSE)
            # print(model, p)
            if p < alpha_value:
                print(f"Model {model} is significantly different to baseline.")
        else:
            print("shapiro", model, p)


def wilcoxon_test(df, alpha_value=0.01, baseline_name="BASELINE"):
    df_baseline = df[df.ModelType == "BASELINE"].sort_values("ParkId")
    for model, df_model in df.groupby("ModelType"):
        if model == baseline_name:
            continue

        df_model = df_model.sort_values("ParkId")

        _, p = stats.wilcoxon(df_baseline.RMSE, df_model.RMSE)

        if p > alpha_value:
            print(f"Model {model} is significantly different to baseline.")


df = get_test_results("./results/test_forecasts/pv/")
# print(df)
t_test(df)

