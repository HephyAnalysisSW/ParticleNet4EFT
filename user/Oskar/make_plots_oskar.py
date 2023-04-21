import sys

sys.path.insert(0, "/users/oskar.rothbacher/CMS/ML-pytorch")
from tools.WeightInfo import WeightInfo

import os
import pathlib
import argparse

import numpy as np
import awkward as ak
import awkward0 as ak0
import uproot
import yaml
import itertools

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages


BASE_DIR = pathlib.Path("/users/oskar.rothbacher/CMS/ParticleNet4EFT").resolve()
USER = os.getlogin()

# DEFAULT_CONFIG = BASE_DIR / "data" / "for_plots" / "wilson_plot.yaml"

DEFAULT_PLOT_DIRECTORY = pathlib.Path("/groups/hephy/cms") / USER / "www/weaver"
DEFAULT_MODEL_PATH = BASE_DIR / "models"

parser = argparse.ArgumentParser()

# parser.add_argument(
#     "-c",
#     "--config-file",
#     default=DEFAULT_CONFIG,
#     help="""
#     data config YAML file, relative path to this script or absoulte path
#     groups: data_path, cut, weights_info, branches: weight_coeff, any other groupings
#     """,
# )
# parser.add_argument(
#     "-n",
#     "--nr-files",
#     default=10,
#     type=int,
#     help="number of files to read",
# )
parser.add_argument(
    "-o",
    "--output",
    type=pathlib.Path,
    default="",
)
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
)
parser.add_argument(
    "--epoch",
    type=str,
    default="last",
)
# parser.add_argument(
#     "--interactive",
#     action="store_true",
# )

args = parser.parse_args()


def main(args):

    args.output = DEFAULT_PLOT_DIRECTORY / args.output
    args.model_name = DEFAULT_MODEL_PATH / args.model_name

    # check if the model path exists
    if args.model_name.exists() is False:
        print(
            f"model folder '{args.model_name}' not found in '{DEFAULT_MODEL_PATH}', quitting."
        )
        sys.exit()

    # check if predict_output exists in the model path
    predict_output_path = args.model_name / "predict_output"
    # print(predict_output_path)
    if predict_output_path.exists():
        pathlib.Path.mkdir(args.output, parents=True, exist_ok=True)
    else:
        print(f"No predictions for {args.model_name} exist.")
        sys.exit()

    # if args.interactive:
    #     global weight_helper, pred_out

    # with open(args.config_file, "r") as c:
    #     config = yaml.safe_load(c)

    # data_files = [config["data_path"].format(i) for i in range(args.nr_files)]
    # branch_grouped_dict = config["branches"]
    # branch_list = list(itertools.chain(*branch_grouped_dict.values()))
    # cut = config.get("cut")

    # if args.nr_files != 0:
    #     data = uproot.concatenate(data_files, branches=branch_list, cut=cut)
    #     weight_data = ak.to_numpy(data[config["branches"]["weight_coeff"][0]])

    # weight_helper = WeightHelper(("ctWRe",), order=2)

    # wSM = weight_helper.make_eft_weights(weight_data, order=0, ctWRe=0)
    # wEFT = weight_helper.make_eft_weights(weight_data, order=2, ctWRe=1)

    # read predict_output root file
    # weaver prediction only supports 1-d arrays as observers when using root files as output format
    # with uproot.open(
    #     str(list(args.model_name.glob("predict_output/*.root"))[-1]) + ":E
    #
    # vents"
    # ) as pred_out_file:
    #     pred_out = pred_out_file.arrays(library="np")

    # read predict_output awkward file
    # weaver supports awkward arrays when using awkward files as output format, usefull for weight data

    # print("predict list")
    # print(str(list(args.model_name.glob("predict_output/*at*.awkd"))[-1]))
    # sys.exit()

    try:
        if args.epoch=="last":
            pred_out = dict(
                ak0.load(str(list(args.model_name.glob("predict_output/*at*.awkd"))[-1]))          # last epoch
            )
        elif args.epoch=="best":
            pred_out = dict(
                ak0.load(args.model_name / "predict_output" / "best_epoch_prediction.awkd")   # 'best' epoch
            )
        else:
            pred_out = dict(
                ak0.load(args.model_name / "predict_output" / f"prediction_at_epoch_{args.epoch}.awkd")   #  epoch
            )
    except:
        sys.exit()

    # print(pred_out.keys())

    lin = pred_out["scores"][:, 0]
    quad = pred_out["scores"][:, 1]
    weights = pred_out["ctWRe_coeffs"]

    # with PdfPages(args.output / "nn_LLR.pdf") as pdf:

    # LLR plot for nn output
    fig = plt.figure(figsize=[12,15], constrained_layout=True)
    fig.suptitle(f"Model evaluation for '{str(args.model_name).split('/')[-1]}'")
    ax_dict = fig.subplot_mosaic(
        """
        AACC
        AACC
        BBCC
        DDEE
        DDEE
        """
    )

    # LLR plot
    interval = plot_LLR_full(
        ax=ax_dict["C"],
        lin=lin,
        quad=quad,
        weights=weights,
        n_plot_grid=100,
        norm_to_nevents=1000
    )

    # interval = np.array([-0.5,0.5])

    # hists
    print("hists")
    plot_nn_hist(
        ax=ax_dict["A"],
        lin=lin,
        quad=quad,
        weights=weights,
        param_values=interval,
        use_weighted_quantiles=True,
        ratio_hist=False,
        norm_to_nevents=1000,
        title="lin+0.5*quad hist",
    )

    print("ratio hists")
    # ratio hists
    plot_nn_hist(
        ax=ax_dict["B"],
        lin=lin,
        quad=quad,
        weights=weights,
        param_values=interval,
        use_weighted_quantiles=True,
        ratio_hist=True,
        norm_to_nevents=1000,
        title="lin+0.5*quad ratio hist"
    )

    # pred target for lin
    pred_target_hist2d(
        ax=ax_dict["D"],
        label="linear coefficient",
        pred=lin,
        target=weights[:, 1],
        bins=100,
        log=True,
    )

    # pred target for quad
    pred_target_hist2d(
        ax=ax_dict["E"],
        label="quadratic coefficient",
        pred=quad,
        target=weights[:, 2],
        bins=100,
        log=True,
    )
    pathlib.Path.mkdir(args.output / f"{str(args.model_name).split('/')[-1]}", parents=True, exist_ok=True)
    fig.savefig(args.output / f"{str(args.model_name).split('/')[-1]}" / f"epoch_{args.epoch}_LLR.png")

    print(args.output)
    # all the plots go inside this context manager
    # with PdfPages(args.output / "nn_LLR.pdf") as pdf:

    #     # LLR plot for nn output
    #     ax_dict = plt.figure(constrained_layout=True).subplot_mosaic(
    #         """
    #         AADD
    #         AADD
    #         BBDD
    #         """
    #     )

    #     # nn hists
    #     # ctWRe = -1
    #     weight_helper.set_eft_params(ctWRe=-1)
    #     plot_hist(
    #         ax=ax_dict["A"],
    #         var_name="scores",
    #         pred_out=pred_out,
    #         weight_helper=weight_helper,
    #         eq_width=False,
    #         ratio_hist=False,
    #     )
    #     # ctWRe = 1
    #     weight_helper.set_eft_params(ctWRe=1)
    #     plot_hist(
    #         ax=ax_dict["A"],
    #         var_name="scores",
    #         pred_out=pred_out,
    #         weight_helper=weight_helper,
    #         eq_width=False,
    #         ratio_hist=False,
    #     )

    #     # nn ratio hists
    #     # ctWRe = -1
    #     weight_helper.set_eft_params(ctWRe=-1)
    #     plot_hist(
    #         ax=ax_dict["B"],
    #         var_name="scores",
    #         pred_out=pred_out,
    #         weight_helper=weight_helper,
    #         eq_width=False,
    #         ratio_hist=True,
    #     )
    #     # ctWRe = 1
    #     weight_helper.set_eft_params(ctWRe=1)
    #     plot_hist(
    #         ax=ax_dict["B"],
    #         var_name="scores",
    #         pred_out=pred_out,
    #         weight_helper=weight_helper,
    #         eq_width=False,
    #         ratio_hist=True,
    #     )

    #     plot_LLR(
    #         ax=ax_dict["D"],
    #         var_names=("scores", "lin_ctWRe"),
    #         eft_param="ctWRe",
    #         pred_out=pred_out,
    #         weight_helper=weight_helper,
    #         n_plot_grid=1000,
    #     )
    #     pdf.savefig()
    #     plt.close()


class WeightHelper:
    def __init__(
        self,
        coefficients: tuple[str, ...] = ("ctWRe",),
        order=2,
        reweight_pkl="/eos/vbc/group/cms/robert.schoefbeck/gridpacks/ParticleNet/t-sch-RefPoint-noWidthRW_reweight_card.pkl",
    ):
        self.coefficients = coefficients
        self.order = order
        self.reweight_pkl = reweight_pkl
        self.eft_params = dict((coeff, 0) for coeff in self.coefficients)

        self.__weight_info = WeightInfo(self.reweight_pkl)
        self.__weight_info.set_order(self.order)

        self.combinations = self.__make_combinations()
        self.indices = self.__get_indices()

    def set_eft_params(self, reset=False, **kwargs):
        assert set(kwargs.keys()).issubset(
            self.eft_params.keys()
        ), f"parameters to set must be in {set(self.eft_params.keys())}"

        if reset:
            self.eft_params = dict((coeff, 0) for coeff in self.coefficients)

        self.eft_params.update(kwargs)

    def make_sm_weights(self, weight_data):
        return weight_data[:, 0]

    def make_eft_weights(self, weight_data, order=None, reset_after=True, **kwargs):
        if order is None:
            order = self.order

        if reset_after:
            current_eft_params = self.eft_params

        if kwargs:
            self.set_eft_params(reset=True, **kwargs)

        eft_weights = 0
        for ind, comb in zip(self.indices, self.combinations):
            if len(comb) <= order:
                eft_weights += (
                    (1 / np.math.factorial(len(comb)))
                    * np.prod([self.eft_params[coeff] for coeff in comb])
                    * weight_data[:, ind]
                )

        if reset_after:
            self.eft_params = current_eft_params

        return eft_weights

    def __make_combinations(self):
        combinations = []
        for comb in self.__weight_info.combinations:
            good = True
            for k in comb:
                if k not in self.coefficients:
                    good = False
                    break
            if good:
                combinations.append(comb)
        return combinations

    def __get_indices(self):
        self.__weight_info.data
        return [
            self.__weight_info.combinations.index(comb) for comb in self.combinations
        ]


# LLR
def LLR(
    variable: np.ndarray,
    weight_data: np.ndarray,
    weight_helper: WeightHelper,
    eft_param: str,
    param_values: np.ndarray,
    bins: int = 20,
    use_weighted_quantiles: bool = True,
    norm_to_nevents: int = 1000,
):
    w_SM = weight_helper.make_sm_weights(weight_data)

    if use_weighted_quantiles:
        bins = weighted_quantile(
            values=variable,
            quantiles=np.linspace(0, 1, bins + 1),
            sample_weight=w_SM,
        )

    n_hat_0, _ = np.histogram(variable, bins=bins, weights=w_SM)
    n_hat_0 = normalize_hist(n_hat_0, norm_to_nevents)

    LLR = []
    for value in param_values:
        n_hat, _ = np.histogram(
            variable,
            bins=bins,
            weights=weight_helper.make_eft_weights(weight_data, **{eft_param: value}),
        )
        n_hat = normalize_hist(n_hat, norm_to_nevents)
        LLR.append(-2 * (n_hat_0 * np.log(n_hat) - n_hat).sum())
    return np.array(LLR)


def plot_LLR(
    ax: plt.Axes,
    var_names: tuple[str, ...],
    eft_param: str,
    pred_out: dict[str, np.ndarray],
    weight_helper: WeightHelper,
    param_range: tuple[float, float] = (-1.0, 1.0),
    n_plot_grid: int = 100,
    bins: int = 20,
    use_weighted_quantiles: bool = True,
    norm_to_nevents: int = 1000,
):
    """test"""
    assert set(var_names).issubset(
        pred_out.keys()
    ), f"the values for'var_names' must be in {pred_out.keys()}"

    plot_grid = np.linspace(*param_range, n_plot_grid)

    LLR_plot = {}

    for var_name in var_names:
        LLR_plot[var_name] = LLR(
            variable=pred_out[var_name],
            weight_data=pred_out["p_C"],
            weight_helper=weight_helper,
            eft_param=eft_param,
            param_values=plot_grid,
            bins=bins,
            use_weighted_quantiles=use_weighted_quantiles,
            norm_to_nevents=norm_to_nevents,
        )

    LLR_min = np.min(LLR_plot[var_names[0]])
    idx_min = np.argwhere(np.diff(np.sign(LLR_plot[var_names[0]] - LLR_min - 1)))

    for var_name in var_names:
        ax.plot(plot_grid, LLR_plot[var_name], label=var_name)

    ax.set_xlabel(eft_param)
    ax.set_ylabel("LLR")

    ax.plot(plot_grid, np.full_like(plot_grid, LLR_min + 1), "k:")
    ax.plot(plot_grid[idx_min], LLR_plot[var_names[0]][idx_min], "or")

    ax.set_title("LLR plot")
    ax.legend()


# LLR from predicted lin and quad weight coeffs
def LLR_full(
    lin: np.ndarray,
    quad: np.ndarray,
    weights: np.ndarray,
    param_values: np.ndarray,
    bins: int = 20,
    use_weighted_quantiles: bool = True,
    norm_to_nevents: int = 1000,
):
    LLR = []
    for value in param_values:
        estimator = lin + 0.5 * value * quad
        w_SM = weights[:, 0]
        w_EFT = weights[:, 0] * (
            1 + value * weights[:, 1] + 0.5 * value**2 * weights[:, 2]
        )

        if use_weighted_quantiles:
            bin_edges = weighted_quantile(
                values=estimator,
                quantiles=np.linspace(0, 1, bins + 1),
                sample_weight=w_SM,
            )
        else:
            bin_edges = bins

        n_hat_0, bin_edges = np.histogram(estimator, bin_edges, weights=w_SM)
        n_hat, _ = np.histogram(estimator, bin_edges, weights=w_EFT)

        n_hat_0 = normalize_hist(n_hat_0, norm_to_nevents)
        n_hat = normalize_hist(n_hat, norm_to_nevents)

        LLR.append(-2 * (n_hat_0 * np.log(n_hat) - n_hat).sum())
    return np.array(LLR)


# LLR plot for full estimator
def plot_LLR_full(
    ax: plt.Axes,
    lin: np.ndarray,
    quad: np.ndarray,
    weights: np.ndarray,
    param_range: tuple[float, float] = (-1.0, 1.0),
    n_plot_grid: int = 100,
    bins: int = 20,
    use_weighted_quantiles: bool = True,
    norm_to_nevents: int = 1000,
):
    plot_grid = np.linspace(*param_range, n_plot_grid)

    LLR_plot = LLR_full(
        lin=lin,
        quad=quad,
        weights=weights,
        param_values=plot_grid,
        bins=bins,
        use_weighted_quantiles=use_weighted_quantiles,
        norm_to_nevents=norm_to_nevents,
    )

    LLR_plot_true = LLR_full(
        lin=weights[:, 1],
        quad=weights[:, 2],
        weights=weights,
        param_values=plot_grid,
        bins=bins,
        use_weighted_quantiles=use_weighted_quantiles,
        norm_to_nevents=norm_to_nevents,
    )

    LLR_min = np.min(LLR_plot)
    idcs_intervall = np.argwhere(np.diff(np.sign(LLR_plot - LLR_min - 1)))

    LLR_plot -= LLR_min
    LLR_plot_true -= LLR_min

    ax.plot(plot_grid, LLR_plot, label="network")
    ax.plot(plot_grid, LLR_plot_true, label="true_weight_coeffs")

    ax.set_xlabel("ctWRe", fontsize="x-small")
    ax.set_ylabel("LLR", fontsize="x-small")

    ax.plot(plot_grid, np.full_like(plot_grid, 1), "k:")
    ax.plot(plot_grid[idcs_intervall], LLR_plot[idcs_intervall], "or")

    ax.set_title("LLR plot", fontsize="small")
    ax.legend(fontsize="x-small")

    ax.tick_params("both", labelsize="xx-small")
    # ax.tick_params("y", labelrotation=90)

    return plot_grid[idcs_intervall].flatten()


# histograms
def plot_hist(
    ax: plt.Axes,
    var_name: str,
    pred_out: dict[str, np.ndarray],
    weight_helper: WeightHelper,
    bins: int = 20,
    eq_width: bool = True,
    ratio_hist: bool = False,
    norm_to_nevents: int = 1000,
):
    w_SM = weight_helper.make_sm_weights(pred_out["p_C"])
    w_EFT = weight_helper.make_eft_weights(pred_out["p_C"])

    if eq_width is not True:
        bins = weighted_quantile(
            pred_out[var_name],
            np.linspace(0, 1, bins + 1),
            sample_weight=w_SM,
        )

    hist_SM, bins = np.histogram(pred_out[var_name], bins, weights=w_SM)
    hist_EFT, _ = np.histogram(pred_out[var_name], bins, weights=w_EFT)
    hist_SM = normalize_hist(hist_SM, norm_to_nevents)
    hist_EFT = normalize_hist(hist_EFT, norm_to_nevents)

    if ratio_hist:
        ratio = hist_SM / hist_EFT
        ax.stairs(ratio, label=f"{weight_helper.eft_params}")
        ax.stairs(np.ones_like(hist_SM))
    else:
        ax.stairs(hist_SM, label=f"{weight_helper.eft_params}")
        ax.stairs(hist_EFT)
    ax.legend()


def plot_nn_hist(
    ax: plt.Axes,
    lin: np.ndarray,
    quad: np.ndarray,
    weights: np.ndarray,
    param_values: np.ndarray,
    bins: int = 20,
    use_weighted_quantiles: bool = True,
    ratio_hist: bool = False,
    norm_to_nevents: int = 1000,
    title: str = None
):
    w_SM = weights[:, 0]

    if use_weighted_quantiles:
        bin_edges = weighted_quantile(
            values=lin,
            quantiles=np.linspace(0, 1, bins + 1),
            sample_weight=w_SM,
        )
    else:
        bin_edges = bins

    n_hat_0, bin_edges = np.histogram(lin, bins=bin_edges, weights=w_SM)
    n_hat_0 = normalize_hist(n_hat_0, norm_to_nevents)

    if ratio_hist:
        ax.stairs(np.ones_like(n_hat_0))
    else:
        ax.stairs(n_hat_0, label="sm")

    for value in param_values:
        estimator = lin + 0.5 * value * quad
        w_SM = weights[:, 0]
        w_EFT = weights[:, 0] * (
            1 + value * weights[:, 1] + 0.5 * value**2 * weights[:, 2]
        )

        if use_weighted_quantiles:
            bin_edges = weighted_quantile(
                values=estimator,
                quantiles=np.linspace(0, 1, bins + 1),
                sample_weight=w_SM,
            )
        else:
            bin_edges = bins

        n_hat_0, bin_edges = np.histogram(estimator, bin_edges, weights=w_SM)
        n_hat, _ = np.histogram(estimator, bin_edges, weights=w_EFT)

        n_hat_0 = normalize_hist(n_hat_0, norm_to_nevents)
        n_hat = normalize_hist(n_hat, norm_to_nevents)

        if ratio_hist:
            ratio = n_hat_0 / n_hat
            ax.stairs(ratio, label=f"ctWRe={value:.2}")
            ax.set_ylim(0.7,1.3)
        else:
            ax.stairs(n_hat, label=f"ctWRe={value:.2}")

    ax.legend(loc="lower right", fontsize="x-small")
    ax.tick_params("both", labelsize="xx-small")
    ax.set_title(title, fontsize="small")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)



def pred_target_hist2d(
    ax: plt.Axes,
    label: str,
    pred: np.ndarray,
    target: np.ndarray,
    bins: int = 100,
    range: list[list[float]] = [[-0.5, 0.5], [-0.5, 0.5]],
    log: bool = True,
):
    ax.hist2d(
        target,
        pred,
        bins=bins,
        range=range,
        norm=colors.SymLogNorm(1) if log else None,
    )
    ax.plot(*range)
    #ax.colorbar() need to figure out how to use with Axes style
    ax.set_title(label, fontsize="small")
    ax.set_xlabel("target", fontsize="x-small")
    ax.set_ylabel("prediction", fontsize="x-small")
    ax.tick_params("both", labelsize="xx-small")
    ax.tick_params("y", labelrotation=90)



def normalize_hist(hist: np.ndarray, norm_to_nevents: int = 1000) -> None:
    """
    normalize a histogram to a number of events
    """
    norm = float(hist.sum())
    hist = hist * (norm_to_nevents / norm)
    return hist


# https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


# def pred_target_hists(model_path, bins=100, range=[[-1, 1], [-1, 1]], log=False):
#     predict_output_files = [
#         file
#         for file in os.listdir(
#             os.path.join(
#                 "/users/oskar.rothbacher/CMS/ParticleNet4EFT/models/",
#                 model_path,
#                 "predict_output",
#             )
#         )
#     ]
#     # print(*predict_output_files, sep='\n')
#     n_plots = int(np.ceil(np.sqrt(len(predict_output_files))))
#     plt.subplots(figsize=[12 * n_plots, 10 * n_plots])
#     for n, file in enumerate(predict_output_files):
#         with uproot.open(
#             os.path.join(
#                 "/users/oskar.rothbacher/CMS/ParticleNet4EFT/models/",
#                 model_path,
#                 "predict_output",
#                 f"{file}:Events",window title barwindow title bar
#             )
#         ) as f:
#             predict = f.arrays(library="np")
#             plt.subplot(n_plots, n_plots, n + 1)
#             hist = plt.hist2d(
#                 predict["lin_ctWRe"],
#                 predict["output"],
#                 bins=bins,
#                 range=range,
#                 norm=colors.SymLogNorm(1) if log else None,
#             )
#             plt.plot([-1, 1], [-1, 1])
#             plt.colorbar()
#             plt.title(file)


# import subprocess


# PATH_TO_DATA = "/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v5/tschRefPointNoWidthRW/"
# DATA_FILES = "tschRefPointNoWidthRW_[8-9]?.root"
# DATA_CONFIG = "data/delphes_hl_features_lin.yaml"
# NETWORK_CONFIG = "mlp_genjetAK8_lin.py"
# MODDEL_PATH = "models/mlp_hl_lin_delphes_test_1/"
# MODEL_PREFIX = "mlp"
# EPOCHS = [399]

# for epoch in EPOCHS:
#     subprocess.run(
#         [
#             "python",
#             "train.py",
#             "--predict",
#             "--data-test",
#             PATH_TO_DATA + DATA_FILES,
#             "--data-config",
#             DATA_CONFIG,
#             "--network-config",
#             "networks/" + NETWORK_CONFIG,
#             "--model-prefix",
#             MODDEL_PATH + MODEL_PREFIX + f"_epoch-{epoch}_state.pt",
#             "--predict-output",
#             f"prediction_at_epoch_{epoch}.root",
#             "--regression-mode",
#             "--gpus",
#             "0",
#         ]
#     )


if __name__ == "__main__":
    main(args=args)


# args = parser.parse_args()

# args.output = DEFAULT_PLOT_DIRECTORY / args.output
# args.model_name = DEFAULT_MODEL_PATH / args.model_name


# if args.model_name.exists() is False:
#     print(
#         f"model folder '{args.model_name}' not found in '{DEFAULT_MODEL_PATH}', quitting."
#     )
#     sys.exit()

# with open(args.config_file, "r") as c:
#     config = yaml.safe_load(c)

# data_files = [config["data_path"].format(i) for i in range(args.nr_files)]
# branch_grouped_dict = config["branches"]
# branch_list = list(itertools.chain(*branch_grouped_dict.values()))
# cut = config.get("cut")

# data = uproot.concatenate(data_files, branches=branch_list, cut=cut)

# weight_data = ak.to_numpy(data[config["branches"]["weight_coeff"][0]])

# weight_helper = WeightHelper(("ctWRe",), order=2)

# wSM = weight_helper.make_eft_weights(weight_data, order=0, ctWRe=0)
# wEFT = weight_helper.make_eft_weights(weight_data, order=2, ctWRe=1)

# print(str(list(args.model_name.glob("predict_output/*.root"))[-1]) + ":Events")
# with uproot.open(
#     str(list(args.model_name.glob("predict_output/*.root"))[-1]) + ":Events"
# ) as pred_out:
#     pred_out.show()
#     arrays = pred_out.arrays(library="np")
#     print(arrays["output"].shape)
#     print(pred_out["output"].array())

# print(weight_data.shape)
# print(wSM[:5])
# print(wEFT[:5])
