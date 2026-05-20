from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

RC_CONTEXT = {
    "default": {},
    "small": {"font.size": 8, "lines.markersize": 4.0},
    "large": {
        "font.size": 15,
        "figure.figsize": [8, 5],
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
    },
}


class AutoSubplot:
    def __init__(
        self,
        ax_size: Optional[Tuple[float, float]] = None,
        max_rows: int = 3,
        max_cols: int = 4,
        ax_num: Optional[int] = None,
        sharex=False,
        sharey=False,
        rc_context="default",
    ):
        self.rc_context = rc_context
        if ax_num is None:
            ax_num = max_rows * max_cols

        if ax_num >= max_rows * max_cols:
            self.row_num = max_rows
            self.col_num = max_cols
        elif ax_num >= max_cols:
            row_num = round(ax_num // max_cols)
            if ax_num % max_cols > 0:
                row_num += 1
            self.row_num = row_num
            self.col_num = max_cols
        else:
            self.row_num = 1
            self.col_num = ax_num

        if ax_size is None:
            ax_size = (3, 2.5)
        self.figsize = [ax_size[0] * self.col_num, ax_size[1] * self.row_num]

        self.figures: List[Figure] = []
        self.axes: List[Axes] = []
        self.current_fig = None
        self.ax_count = 0
        self.sharex = sharex
        self.sharey = sharey

    def add_subplot(self) -> Axes:
        with plt.rc_context(RC_CONTEXT[self.rc_context]):
            if (
                self.current_fig is None
                or self.ax_count % (self.row_num * self.col_num) == 0
            ):
                self.current_fig = plt.figure(figsize=self.figsize)
                self.figures.append(self.current_fig)
            sharex = None
            sharey = None
            if len(self.axes) > 0:
                if self.sharex:
                    sharex = self.axes[0]
                if self.sharey:
                    sharey = self.axes[0]
            ax = self.current_fig.add_subplot(
                self.row_num,
                self.col_num,
                (self.ax_count % (self.row_num * self.col_num)) + 1,
                sharex=sharex,
                sharey=sharey,
            )
        self.axes.append(ax)
        self.ax_count += 1
        return ax

    def xlabel(self, xlabel: str):
        with plt.rc_context(RC_CONTEXT[self.rc_context]):
            for fig in self.figures:
                plt.figure(fig)
                axes = fig.axes
                if len(axes) > self.col_num:
                    axes = axes[-self.col_num :]
                for ax in axes:
                    plt.axes(ax)
                    plt.xlabel(xlabel)

    def ylabel(self, ylabel: str):
        with plt.rc_context(RC_CONTEXT[self.rc_context]):
            for fig in self.figures:
                plt.figure(fig)
                axes = fig.axes
                for idx, ax in enumerate(axes):
                    if idx % (self.col_num) == 0:
                        plt.axes(ax)
                        plt.ylabel(ylabel)

    def xlim(self, xlim):
        for fig in self.figures:
            plt.figure(fig)
            axes = fig.axes
            for ax in axes:
                plt.axes(ax)
                plt.xlim(xlim)

    def ylim(self, ylim):
        for fig in self.figures:
            plt.figure(fig)
            axes = fig.axes
            for ax in axes:
                plt.axes(ax)
                plt.ylim(ylim)

    def suptitle(self, title: str):
        with plt.rc_context(RC_CONTEXT[self.rc_context]):
            for fig in self.figures:
                fig.suptitle(title)

    def tight_layout(self):
        for fig in self.figures:
            fig.tight_layout()
