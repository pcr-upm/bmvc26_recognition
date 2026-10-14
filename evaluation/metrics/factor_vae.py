# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.

"""Implementation of the FactorVAE disentanglement metric.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).

Modified compact diagnostic version:
  - Keeps the original FactorVAE score unchanged.
  - Stores a compact training vote matrix: projected_unit x ground_truth_factor.
  - Stores the final train-time assignment of each projected unit.
  - Stores how many projected units are assigned to each ground-truth factor.
  - Stores a compact eval-time stability diagnostic:
      mean normalized variance of each unit when its assigned factor is fixed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from evaluation.metrics import utils
import numpy as np
from six.moves import range
import gin.tf


@gin.configurable(
    "factor_vae_score",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_factor_vae(ground_truth_data,
                       representation_function,
                       random_state,
                       artifact_dir=None,
                       batch_size=gin.REQUIRED,
                       num_train=gin.REQUIRED,
                       num_eval=gin.REQUIRED,
                       num_variance_estimate=gin.REQUIRED):
  """Computes the FactorVAE disentanglement metric.

  Returns:
    Dictionary with the original FactorVAE scores plus compact vote diagnostics.
  """
  del artifact_dir

  logging.info("Computing global variances to standardise.")
  global_variances = _compute_variances(
      ground_truth_data,
      representation_function,
      num_variance_estimate,
      random_state)

  active_dims = _prune_dims(global_variances)
  scores_dict = {}

  if not active_dims.any():
    scores_dict["train_accuracy"] = 0.0
    scores_dict["eval_accuracy"] = 0.0
    scores_dict["num_active_dims"] = 0
    scores_dict["num_total_dims"] = int(global_variances.shape[0])
    scores_dict["active_dim_indices"] = []
    scores_dict["factor_vae_vote_diagnostics"] = {
        "metadata": {
            "num_factors": int(ground_truth_data.num_factors),
            "num_projected_units": int(global_variances.shape[0]),
            "num_train_votes": int(num_train),
            "num_eval_votes": int(num_eval),
            "batch_size_per_vote": int(batch_size),
            "num_variance_estimate": int(num_variance_estimate),
            "factor_names": _get_factor_names(ground_truth_data),
            "note": "No active dimensions were found after pruning.",
        },
        "train_votes_unit_by_factor": [],
        "unit_assignments_train": [],
        "factor_assignment_counts_train": [],
        "eval_assigned_factor_stability": [],
    }
    return scores_dict

  logging.info("Generating training votes.")
  training_votes, _, _ = _generate_training_batch(
      ground_truth_data=ground_truth_data,
      representation_function=representation_function,
      batch_size=batch_size,
      num_points=num_train,
      random_state=random_state,
      global_variances=global_variances,
      active_dims=active_dims,
      collect_variance_stats=False)

  # classifier[j] is the GT factor assigned to projected latent dimension j.
  classifier = np.argmax(training_votes, axis=0)
  other_index = np.arange(training_votes.shape[1])

  logging.info("Evaluating training accuracy.")
  train_denominator = np.sum(training_votes)
  if train_denominator == 0:
    train_accuracy = 0.0
  else:
    train_accuracy = (
        np.sum(training_votes[classifier, other_index]) * 1.0
        / train_denominator)

  logging.info("Training set accuracy: %.2g", train_accuracy)

  logging.info("Generating evaluation votes.")
  eval_votes, eval_mean_normvar_factor_by_unit, eval_factor_counts = (
      _generate_training_batch(
          ground_truth_data=ground_truth_data,
          representation_function=representation_function,
          batch_size=batch_size,
          num_points=num_eval,
          random_state=random_state,
          global_variances=global_variances,
          active_dims=active_dims,
          collect_variance_stats=True))

  logging.info("Evaluating evaluation accuracy.")
  eval_denominator = np.sum(eval_votes)
  if eval_denominator == 0:
    eval_accuracy = 0.0
  else:
    eval_accuracy = (
        np.sum(eval_votes[classifier, other_index]) * 1.0
        / eval_denominator)

  logging.info("Evaluation set accuracy: %.2g", eval_accuracy)

  scores_dict["train_accuracy"] = float(train_accuracy)
  scores_dict["eval_accuracy"] = float(eval_accuracy)

  # More informative than len(active_dims), which is the total number of dims.
  scores_dict["num_active_dims"] = int(np.sum(active_dims))
  scores_dict["num_total_dims"] = int(global_variances.shape[0])
  scores_dict["active_dim_indices"] = [
      int(i) for i in np.where(active_dims)[0]
  ]

  scores_dict["factor_vae_vote_diagnostics"] = _build_compact_vote_diagnostics(
      ground_truth_data=ground_truth_data,
      active_dims=active_dims,
      training_votes=training_votes,
      eval_votes=eval_votes,
      eval_mean_normvar_factor_by_unit=eval_mean_normvar_factor_by_unit,
      eval_factor_counts=eval_factor_counts,
      classifier=classifier,
      batch_size=batch_size,
      num_train=num_train,
      num_eval=num_eval,
      num_variance_estimate=num_variance_estimate)

  return scores_dict


@gin.configurable("prune_dims", blacklist=["variances"])
def _prune_dims(variances, threshold=0.):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold


def _compute_variances(ground_truth_data,
                       representation_function,
                       batch_size,
                       random_state,
                       eval_batch_size=64):
  """Computes the global variance for each representation dimension."""
  observations = ground_truth_data.sample_observations(batch_size, random_state)
  representations = utils.obtain_representation(
      observations,
      representation_function,
      eval_batch_size)
  representations = np.transpose(representations)
  assert representations.shape[0] == batch_size
  return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(ground_truth_data,
                              representation_function,
                              batch_size,
                              random_state,
                              global_variances,
                              active_dims):
  """Samples one FactorVAE vote.

  Returns:
    factor_index:
      Ground-truth factor fixed in this mini-batch.

    argmin:
      Original projected-dimension index with minimum normalized variance.

    normalized_variances:
      local_variance / global_variance for every projected dimension.
  """
  factor_index = random_state.randint(ground_truth_data.num_factors)

  factors = ground_truth_data.sample_factors(batch_size, random_state)
  factors[:, factor_index] = factors[0, factor_index]

  observations = ground_truth_data.sample_observations_from_factors(
      factors,
      random_state)

  representations = representation_function(observations)
  local_variances = np.var(representations, axis=0, ddof=1)

  normalized_variances = _safe_normalized_variances(
      local_variances,
      global_variances)

  active_indices = np.where(active_dims)[0]
  argmin_active = np.argmin(normalized_variances[active_dims])
  argmin = active_indices[argmin_active]

  return int(factor_index), int(argmin), normalized_variances


def _generate_training_batch(ground_truth_data,
                             representation_function,
                             batch_size,
                             num_points,
                             random_state,
                             global_variances,
                             active_dims,
                             collect_variance_stats=False):
  """Samples a batch of FactorVAE votes.

  Returns:
    votes:
      Shape (num_factors, num_projected_units).
      votes[f, u] is how many times projected unit u voted for GT factor f.

    mean_normvar_factor_by_unit:
      Shape (num_factors, num_projected_units), only if collect_variance_stats.
      Entry [f, u] is the mean normalized variance of unit u over mini-batches
      where factor f was fixed.

    factor_counts:
      Number of mini-batches sampled for each fixed factor.
  """
  num_factors = ground_truth_data.num_factors
  num_units = global_variances.shape[0]

  votes = np.zeros((num_factors, num_units), dtype=np.int64)

  if collect_variance_stats:
    normvar_sum = np.zeros((num_factors, num_units), dtype=np.float64)
    factor_counts = np.zeros(num_factors, dtype=np.int64)
  else:
    normvar_sum = None
    factor_counts = None

  for _ in range(num_points):
    factor_index, argmin, normalized_variances = _generate_training_sample(
        ground_truth_data=ground_truth_data,
        representation_function=representation_function,
        batch_size=batch_size,
        random_state=random_state,
        global_variances=global_variances,
        active_dims=active_dims)

    votes[factor_index, argmin] += 1

    if collect_variance_stats:
      normvar_sum[factor_index, :] += normalized_variances
      factor_counts[factor_index] += 1

  if collect_variance_stats:
    mean_normvar_factor_by_unit = _mean_from_sum_and_counts(
        normvar_sum,
        factor_counts)
  else:
    mean_normvar_factor_by_unit = None

  return votes, mean_normvar_factor_by_unit, factor_counts


def _build_compact_vote_diagnostics(ground_truth_data,
                                    active_dims,
                                    training_votes,
                                    eval_votes,
                                    eval_mean_normvar_factor_by_unit,
                                    eval_factor_counts,
                                    classifier,
                                    batch_size,
                                    num_train,
                                    num_eval,
                                    num_variance_estimate):
  """Builds a compact JSON-serializable diagnostic block."""
  factor_names = _get_factor_names(ground_truth_data)

  training_votes = np.asarray(training_votes, dtype=np.int64)
  eval_votes = np.asarray(eval_votes, dtype=np.int64)
  classifier = np.asarray(classifier, dtype=np.int64)
  active_dims = np.asarray(active_dims).astype(bool)

  num_factors, num_units = training_votes.shape

  unit_assignments = _build_unit_assignments_train(
      training_votes=training_votes,
      classifier=classifier,
      active_dims=active_dims,
      factor_names=factor_names)

  factor_assignment_counts = _build_factor_assignment_counts_train(
      unit_assignments=unit_assignments,
      num_factors=num_factors,
      factor_names=factor_names)

  eval_assigned_factor_stability = _build_eval_assigned_factor_stability(
      unit_assignments=unit_assignments,
      eval_votes=eval_votes,
      eval_mean_normvar_factor_by_unit=eval_mean_normvar_factor_by_unit,
      eval_factor_counts=eval_factor_counts)

  factors_without_assigned_unit = [
      item["factor"]
      for item in factor_assignment_counts
      if item["num_units_assigned"] == 0
  ]

  return {
      "metadata": {
          "num_factors": int(num_factors),
          "num_projected_units": int(num_units),
          "num_train_votes": int(num_train),
          "num_eval_votes": int(num_eval),
          "batch_size_per_vote": int(batch_size),
          "num_variance_estimate": int(num_variance_estimate),
          "factor_names": factor_names,
          "note": (
              "train_votes_unit_by_factor has rows=projected units and "
              "columns=ground-truth factors. Each projected unit corresponds "
              "to one PCA-projected latent unit in the DisDiff evaluation."
          ),
      },

      # Compact raw vote table: rows are units, columns are GT factors.
      "train_votes_unit_by_factor": training_votes.T.astype(int).tolist(),

      # For each projected unit: final factor selected by train-time majority.
      "unit_assignments_train": unit_assignments,

      # For each GT factor: how many units end up assigned to it.
      "factor_assignment_counts_train": factor_assignment_counts,

      # Summary useful to quickly see whether every factor gets at least one unit.
      "num_factors_with_assigned_unit_train": int(
          np.sum([
              item["num_units_assigned"] > 0
              for item in factor_assignment_counts
          ])),
      "factors_without_assigned_unit_train": factors_without_assigned_unit,

      # Compact eval-time check:
      # for each unit, mean normalized variance when its assigned factor is fixed.
      "eval_assigned_factor_stability": eval_assigned_factor_stability,
  }


def _build_unit_assignments_train(training_votes,
                                  classifier,
                                  active_dims,
                                  factor_names):
  """Returns one compact dictionary per projected unit."""
  num_factors, num_units = training_votes.shape

  unit_assignments = []

  for unit_idx in range(num_units):
    column = training_votes[:, unit_idx]
    total_votes = int(np.sum(column))
    is_active = bool(active_dims[unit_idx])

    if is_active and total_votes > 0:
      assigned_factor_idx = int(classifier[unit_idx])
      assigned_factor = factor_names[assigned_factor_idx]
      assigned_votes = int(column[assigned_factor_idx])
      assignment_purity = _safe_divide(assigned_votes, total_votes)
    else:
      assigned_factor_idx = None
      assigned_factor = None
      assigned_votes = 0
      assignment_purity = None

    unit_assignments.append({
        "unit": int(unit_idx),
        "is_active": is_active,
        "assigned_factor_index": assigned_factor_idx,
        "assigned_factor": assigned_factor,
        "assigned_votes": int(assigned_votes),
        "total_votes_received": int(total_votes),
        "assignment_purity": _safe_float_or_none(assignment_purity),
    })

  return unit_assignments


def _build_factor_assignment_counts_train(unit_assignments,
                                          num_factors,
                                          factor_names):
  """Counts how many projected units are assigned to each GT factor."""
  factor_assignment_counts = []

  for factor_idx in range(num_factors):
    assigned_units = []
    purities = []
    total_votes_from_assigned_units = 0

    for item in unit_assignments:
      if item["assigned_factor_index"] == factor_idx:
        assigned_units.append(int(item["unit"]))
        total_votes_from_assigned_units += int(item["assigned_votes"])
        if item["assignment_purity"] is not None:
          purities.append(float(item["assignment_purity"]))

    if len(purities) > 0:
      mean_assignment_purity = float(np.mean(purities))
    else:
      mean_assignment_purity = None

    factor_assignment_counts.append({
        "factor_index": int(factor_idx),
        "factor": factor_names[factor_idx],
        "num_units_assigned": int(len(assigned_units)),
        "assigned_units": assigned_units,
        "total_train_votes_from_assigned_units": int(
            total_votes_from_assigned_units),
        "mean_assignment_purity": _safe_float_or_none(mean_assignment_purity),
    })

  return factor_assignment_counts


def _build_eval_assigned_factor_stability(unit_assignments,
                                          eval_votes,
                                          eval_mean_normvar_factor_by_unit,
                                          eval_factor_counts):
  """Builds compact eval stability information for each assigned unit."""
  stability = []

  for item in unit_assignments:
    unit_idx = item["unit"]
    assigned_factor_idx = item["assigned_factor_index"]

    if assigned_factor_idx is None:
      stability.append({
          "unit": int(unit_idx),
          "assigned_factor_index": None,
          "assigned_factor": None,
          "eval_votes_for_assigned_factor": None,
          "eval_batches_where_assigned_factor_was_fixed": None,
          "eval_mean_normalized_variance_when_assigned_factor_fixed": None,
      })
      continue

    mean_normvar = eval_mean_normvar_factor_by_unit[
        assigned_factor_idx,
        unit_idx]

    stability.append({
        "unit": int(unit_idx),
        "assigned_factor_index": int(assigned_factor_idx),
        "assigned_factor": item["assigned_factor"],
        "eval_votes_for_assigned_factor": int(
            eval_votes[assigned_factor_idx, unit_idx]),
        "eval_batches_where_assigned_factor_was_fixed": int(
            eval_factor_counts[assigned_factor_idx]),
        "eval_mean_normalized_variance_when_assigned_factor_fixed":
            _safe_float_or_none(mean_normvar),
    })

  return stability


def _safe_normalized_variances(local_variances, global_variances):
  """Computes local_variance / global_variance safely."""
  local_variances = np.asarray(local_variances, dtype=np.float64)
  global_variances = np.asarray(global_variances, dtype=np.float64)

  normalized = np.full_like(local_variances, np.inf, dtype=np.float64)

  valid = global_variances > 0.0
  normalized[valid] = local_variances[valid] / global_variances[valid]

  return normalized


def _mean_from_sum_and_counts(values_sum, counts):
  """Computes mean values per factor, returning NaN where count is zero."""
  values_sum = np.asarray(values_sum, dtype=np.float64)
  counts = np.asarray(counts, dtype=np.int64)

  means = np.full_like(values_sum, np.nan, dtype=np.float64)

  for factor_idx in range(values_sum.shape[0]):
    if counts[factor_idx] > 0:
      means[factor_idx, :] = values_sum[factor_idx, :] / float(
          counts[factor_idx])

  return means


def _safe_divide(numerator, denominator):
  if denominator == 0:
    return 0.0
  return float(numerator) / float(denominator)


def _safe_float_or_none(value):
  if value is None:
    return None

  value = float(value)

  if not np.isfinite(value):
    return None

  return value


def _get_factor_names(ground_truth_data):
  """Tries to recover factor names; otherwise returns factor_0, factor_1, ..."""
  candidate_attrs = [
      "factor_names",
      "factors_names",
      "ground_truth_factor_names",
  ]

  for attr in candidate_attrs:
    if hasattr(ground_truth_data, attr):
      names = getattr(ground_truth_data, attr)

      if callable(names):
        try:
          names = names()
        except TypeError:
          names = None

      if names is not None:
        try:
          names = list(names)
          if len(names) == int(ground_truth_data.num_factors):
            return [str(name) for name in names]
        except TypeError:
          pass

  if hasattr(ground_truth_data, "state_space"):
    state_space = getattr(ground_truth_data, "state_space")
    if hasattr(state_space, "factor_names"):
      names = getattr(state_space, "factor_names")

      if callable(names):
        try:
          names = names()
        except TypeError:
          names = None

      if names is not None:
        try:
          names = list(names)
          if len(names) == int(ground_truth_data.num_factors):
            return [str(name) for name in names]
        except TypeError:
          pass

  return [
      "factor_{}".format(i)
      for i in range(int(ground_truth_data.num_factors))
  ]