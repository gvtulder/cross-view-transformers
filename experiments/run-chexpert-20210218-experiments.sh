# exit on errors
set -e

# get the current git commmit
git_commit_hash=$( git rev-parse --short HEAD )

counter=0

cat <<EOF
#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --account=gvtulder
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/gvtulder/ru/transformer/slurm-logs/%x-%A.%a.out
#SBATCH -e /home/gvtulder/ru/transformer/slurm-logs/%x-%A.%a.err
#SBATCH --array=1-1  # TODO
#SBATCH --mem=10G
exit

source /home/gvtulder/ru/venv/bin/activate

EOF

# runs an experiment if the checkpoint file best_val_accuracy.tar does not already exist
function run_train () {
  experiment_id=$1
  shift

  # run only a subset
  if [[ $experiment_id =~ ln-add-linear-do ]] && [[ $experiment_id =~ bidirectional ]] ; then

  tensorboard_dir=logs/log-chexpert-20210218/$experiment_id
  checkpoints_dir=checkpoints/checkpoints-chexpert-20210218/$experiment_id
  if [[ -f $checkpoints_dir/best_val_accuracy.tar ]] ; then
    echo "# Experiment $experiment_id already completed."
  else
    echo "# Starting $experiment_id"
    counter=$((counter + 1))
    echo "if [[ \$SLURM_ARRAY_TASK_ID == $counter ]] ; then"
    cat << EOF
    python -u train.py \\
      --git-commit $git_commit_hash \\
      --device cuda \\
      --autocast \\
      --num-workers 5 \\
      --epochs 60 \\
      --mb-size 32 \\
      --lr-schedule CosineAnnealingLR \\
      --lr 0.00001 \\
      --lr-schedule-eta-min 0.0000001 \\
      --classes 3 \
      --tasks 14 \
      --data CheXpertDataset \
      --train-data data/chexpert-small-20210209-custom-train.h5 \
      --val-data data/chexpert-small-20210209-custom-val.h5 \
      --test-data data/chexpert-small-20210209-custom-test.h5 \
      --augment flip elastic crop20 \\
      $@ \\
      --tensorboard-dir $tensorboard_dir \\
      --checkpoints-dir $checkpoints_dir \\
      --best-checkpoints-only
EOF
    echo "fi"
    echo
  fi
  fi
}


for run in 1 2 3 4 5 ; do

  prefix="20210218"
  postfix="run$run-aug-cosine0.00001-epochs60"

  # models with a single view
  for model in SingleViewResNet18ShallowTop ; do
    for view in frontal lateral; do
      experiment_id="$prefix-model-$model-data-chexpert-view-$view-$postfix"
      run_train $experiment_id \
        --model $model \
        --chexpert-views $view
    done
  done

  # model with both views
  model=LateJoinResNet18ShallowTop
  experiment_id="$prefix-model-$model-data-chexpert-view-ccmlo-$postfix"
  run_train $experiment_id \
    --model $model \
    --chexpert-views frontal lateral

  # model with both views and attention,
  # run for both directions and for different combination methods
  for view in frontallateral; do
    if [[ "$view" == "frontallateral" ]] ; then
      view_list="frontal lateral"
    else
      view_list="lateral frontal"
    fi
    for model in TwoViewAttentionResNet18ShallowTop ; do
      for comb in add-linear ln-add-linear-do ; do
        for heads in 6 12 ; do
          for l1loss in 0.0 ; do
            for entmax in softmax ; do
              for layers in 3 ; do
                for tokens in 16 32 ; do
                  tokenlayers=3
                  experiment_id="$prefix-model-$model-attnheads-$heads-attndds-1-attncomb-$comb-attnl1-$l1loss-$entmax-attntokens-$tokens-attntokenlayers-$tokenlayers-data-chexpert-view-$view-$postfix"
                  run_train $experiment_id \
                    --model $model \
                    --attention-heads $heads \
                    --attention-downsampling 1 \
                    --attention-l1-loss $l1loss \
                    --attention-combine $comb \
                    --attention-tokens $tokens \
                    --attention-token-layers $tokenlayers \
                    --attention-implementation traditional \
                    --chexpert-views $view_list

                  experiment_id="$prefix-model-$model-attnheads-$heads-attndds-1-attncomb-$comb-attnl1-$l1loss-$entmax-attntokens-$tokens-attntokenlayers-$tokenlayers-attnbidirectional-data-chexpert-view-$view-$postfix"
                  run_train $experiment_id \
                    --model $model \
                    --attention-heads $heads \
                    --attention-downsampling 1 \
                    --attention-l1-loss $l1loss \
                    --attention-combine $comb \
                    --attention-tokens $tokens \
                    --attention-token-layers $tokenlayers \
                    --attention-implementation traditional \
                    --attention-bidirectional \
                    --chexpert-views $view_list
                done
              done
            done
          done
        done
      done
    done

    for model in TwoViewAttentionLevel2ResNet18ShallowTop ; do
      comb=add-linear
      for heads in 12 ; do
        for l1loss in 0.0 ; do
          for entmax in softmax ; do
            for layers in 3 ; do
              for tokens in 32 ; do
                tokenlayers=3
                experiment_id="$prefix-model-$model-attnheads-$heads-attndds-1-attncomb-$comb-attnl1-$l1loss-$entmax-attntokens-$tokens-attntokenlayers-$tokenlayers-attnbidirectional-data-chexpert-view-$view-$postfix"
                run_train $experiment_id \
                  --model $model \
                  --attention-heads $heads \
                  --attention-downsampling 1 \
                  --attention-l1-loss $l1loss \
                  --attention-combine $comb \
                  --attention-tokens $tokens \
                  --attention-token-layers $tokenlayers \
                  --attention-implementation traditional \
                  --attention-bidirectional \
                  --chexpert-views $view_list
              done
            done
          done
        done
      done
    done
  done
done

echo
echo "#SBATCH --array=1-$counter"
echo
