from seqio import Mixture, ShardInfo
from typing import Mapping, Optional, Sequence, List
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
import os
from tqdm import tqdm


def save_mixture(
    mixture,
    sequence_length: Optional[Mapping[str, int]] = None,
    split: str = "train",
    use_cached: bool = False,
    shuffle: bool = True,
    seed: Optional[int] = None,
    shard_info: Optional[ShardInfo] = None,
    num_epochs: Optional[int] = None,  # Unique default for Mixture
    copy_pretokenized: bool = False,  # Unique (and all below) to Mixture
    compute_stats_empirically: bool = False,
    log_mixing_proportions: bool = True,
    passthrough_features: Optional[Sequence[str]] = None,
    trim_output_features: bool = True,
    root_dir = None,
    max_sample_per_task = None
):
    assert max_sample_per_task is not None and root_dir is not None
    print("Storing {} datasets to {}...".format(len(mixture.tasks), root_dir))
    if not os.path.exists(root_dir):
      os.makedirs(root_dir)
    
    print("Sample per dataset: {}".format(max_sample_per_task))
    
    mixture._check_compatible_features()
    tasks = []
    for task in mixture.tasks:
      if split not in task.splits:
        logging.warning(
            "Task %s has no '%s' split, skipping.", task.name, split
        )
        continue
      tasks.append(task)
    if not tasks:
      raise ValueError("No datasets have a '{}' split".format(split))

    output_feature_keys = set(mixture.output_features.keys())
    if copy_pretokenized:
      output_feature_keys.update(
          {f + "_pretokenized" for f in output_feature_keys}
      )

    if passthrough_features:
      output_feature_keys.update(passthrough_features)

    def filter_features(ex):
      return {k: v for k, v in ex.items() if k in output_feature_keys}

    datasets: List[tf.data.Dataset] = []
    for task in tqdm(tasks):
      save_dir = os.path.join(root_dir, task.name)
      try:
        if os.path.exists(save_dir):
          print(
            "task '%s' already exists for mixture '%s'",
            task.name,
            mixture.name,
            )
          continue
        ds = task.get_dataset(
            sequence_length,
            split=split,
            use_cached=use_cached,
            shuffle=shuffle,
            seed=seed,
            shard_info=shard_info,
            num_epochs=num_epochs,
            trim_output_features=trim_output_features,
        ).map(filter_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.take(max_sample_per_task)
        tf.data.Dataset.save(ds, save_dir)
      except:
        logging.error(
            "Failed to load task '%s' as part of mixture '%s'",
            task.name,
            mixture.name,
        )
        # Re-raise the same exception, same stack-trace.
        continue
    return datasets
