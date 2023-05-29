# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from collections import defaultdict
from pathlib import Path
from typing import Dict, List, OrderedDict, Tuple
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from vlamp.dataset_readers.common import AnnotatedVideo, Step, Task


def create_task_graph_from_annotations(
    annotations: List[AnnotatedVideo], task: Task
) -> nx.DiGraph:
    annotations = [
        a for a in annotations if a.task is not None and task.idx == a.task.idx
    ]

    G = nx.DiGraph()

    def _count_node(step: Step, increase_count: bool = False):
        if step.idx not in G.nodes:
            G.add_node(
                step.idx,
                # TODO: Fix stepid2desc keys to not require int
                desc=task.stepid2desc[step.idx] if task.stepid2desc else "",
                count=1 if increase_count else 0,
            )
        elif increase_count:
            G.nodes[step.idx]["count"] += 1

    def _count_edge(prev: Step, next: Step, increase_count: bool = True):
        if (prev.idx, next.idx) not in G.edges:
            assert prev.idx in G.nodes
            assert next.idx in G.nodes
            G.add_edge(prev.idx, next.idx, count=1)
        elif increase_count:
            G.edges[(prev.idx, next.idx)]["count"] += 1

    for a in annotations:
        for previous_step, next_step in zip(a.steps[:-1], a.steps[1:]):
            _count_node(previous_step, increase_count=True)
            _count_node(next_step)
            _count_edge(previous_step, next_step, increase_count=True)
    # Add transition probabilities to edges
    for head, tail in G.edges:  # type: str, str
        G.edges[(head, tail)]["transition_p"] = (
            G.edges[(head, tail)]["count"] / G.nodes[head]["count"]
        )
    return G


def plot_adjacency_matrix(
    G: nx.DiGraph,
    filepath: Path,
    title: str,
    weight="weight",
):
    M = nx.to_numpy_matrix(G, weight=weight)
    fig, ax = plt.subplots()
    im = ax.imshow(M)

    # Show all ticks and label them with the respective list entries
    labels = [G.nodes[node]["desc"][:20] for node in G.nodes()]
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_hist(
    num_steps_per_vid,
    filepath,
    xlabel: str = "# steps per video",
    ylabel: str = "# videos",
    title: str = "Steps per video",
):
    fig, ax = plt.subplots()
    ax.hist(num_steps_per_vid, 15, density=False, facecolor="g", alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    plt.savefig(filepath)
    plt.close()


def plot_paths(
    annotations: List[AnnotatedVideo],
    task: Task,
    filepath: Path,
    title: str,
    threshold: float = 0.001,
):
    # Create dict where key is a step in time and value is another dict for which the key is an action pair (idx) and value is its count.
    steps: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )
    total_count: int = 0
    # In general, step_ids can by any string.
    # Here we assign increasing integer ids to steps/action starting from 0
    stepids2int: Dict[str, int] = OrderedDict(
        (idx, i) for i, (idx, desc) in enumerate(task.stepid2desc.items())
    )

    for vid in annotations:
        for i, (previous_action, next_action) in enumerate(
            zip(vid.steps[:-1], vid.steps[1:])
        ):
            steps[(i, i + 1)][
                (stepids2int[previous_action.idx], stepids2int[next_action.idx])
            ] += 1
            total_count += 1

    fig, ax = plt.subplots()
    for (x1, x2), counts in steps.items():
        for (y1, y2), count in counts.items():
            alpha = 10 * count / total_count
            if count / total_count > threshold:
                ax.plot(
                    [x1, x2],
                    [
                        y1,
                        y2,
                    ],
                    alpha=alpha if alpha <= 1 else 1,
                    color="red",
                    marker="o",
                )

    # Show all ticks and label them with the respective list entries
    labels = [
        task.stepid2desc[action_id][:20] for action_id, intid in stepids2int.items()
    ]
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.xaxis.get_major_locator().set_params(integer=True)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filepath)
    plt.close()
