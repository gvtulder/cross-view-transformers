# exit on errors
set -e

# get the current git commmit
git_commit_hash=$( git rev-parse --short HEAD )

counter=0

cat <<EOF
#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --account=gvtulder
#SBATCH --time=2:00:00
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
  fold=$2
  shift
  shift

  # run only a subset
  # if [[ $experiment_id =~ ln-add-linear-do ]] && [[ $experiment_id =~ bidirectional ]] ; then
  if [[ $experiment_id =~ repeat ]] ; then

  foldTrainA=$(( (1 + $fold) % 5 ))
  foldTrainB=$(( (2 + $fold) % 5 ))
  foldTrainC=$(( (3 + $fold) % 5 ))
  foldVal=$(( (4 + $fold) % 5 ))
  foldTest=$(( ($fold) % 5 ))

  tensorboard_dir=logs/log-ddsm-20210218-5cv-ds0.125/$experiment_id
  checkpoints_dir=checkpoints/checkpoints-ddsm-20210218-5cv-ds0.125/$experiment_id
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
      --num-workers 4 \\
      --epochs 300 \\
      --mb-size 24 \\
      --lr-schedule CosineAnnealingLR \\
      --lr 0.0001 \\
      --lr-schedule-eta-min 0.000001 \\
      --lr-warmup-epochs 30 \\
      --swa \\
      --swa-start-epoch 250 \\
      --swa-lr 0.000001 \\
      --classes 2 \\
      --weighted-loss \\
      --normalize \\
      --data DDSMDataset \\
      --train-data data/20210219-ddsm-csv-rescale0.5-nyucrop-ds0.125-mass-CC+MLO-subset$foldTrainA.h5 \\
                   data/20210219-ddsm-csv-rescale0.5-nyucrop-ds0.125-mass-CC+MLO-subset$foldTrainB.h5 \\
                   data/20210219-ddsm-csv-rescale0.5-nyucrop-ds0.125-mass-CC+MLO-subset$foldTrainC.h5 \\
      --val-data   data/20210219-ddsm-csv-rescale0.5-nyucrop-ds0.125-mass-CC+MLO-subset$foldVal.h5 \\
      --test-data  data/20210219-ddsm-csv-rescale0.5-nyucrop-ds0.125-mass-CC+MLO-subset$foldTest.h5 \\
      --augment coflip elastic crop20 \\
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


for run in 1 2 3 ; do
  for fold in 0 1 2 3 4 ; do

    prefix="20210218-ds0.125"
    postfix="run$run-fold$fold-augcoflip-weight-cosine0.0001-warmup30-swa"

    # models with a single view
    for model in SingleViewResNet18ShallowTop ; do
      for view in cc mlo ; do
        experiment_id="$prefix-model-$model-data-ddsm-view-$view-$postfix"
        run_train $experiment_id $fold \
          --model $model \
          --ddsm-views $view
      done
    done

    # model with both views
    model=LateJoinResNet18ShallowTop
    experiment_id="$prefix-model-$model-data-ddsm-view-ccmlo-$postfix"
    run_train $experiment_id $fold \
      --model $model \
      --ddsm-views cc mlo

    # repeat the model several times
    for repeat in 1 2 3 4 5 6 ; do
      model=LateJoinResNet18ShallowTop
      experiment_id="$prefix-model-$model-repeat$repeat-data-ddsm-view-ccmlo-$postfix"
      run_train $experiment_id $fold \
        --model $model \
        --ddsm-views cc mlo
    done

    # model with both views and attention,
    # run for both directions and for different combination methods
    for view in ccmlo ; do
      if [[ "$view" == "ccmlo" ]] ; then
        view_list="cc mlo"
      else
        view_list="mlo cc"
      fi
      for model in TwoViewAttentionResNet18ShallowTop ; do
        for comb in add-linear ln-add-linear-do ; do
          for heads in 12 18 24 ; do
            for l1loss in 0.0 ; do
              for entmax in softmax ; do
                for tokenlayers in 3 ; do
                  for tokens in 16 32 48 64 ; do
                    experiment_id="$prefix-model-$model-attnheads-$heads-attndds-1-attncomb-$comb-attnl1-$l1loss-$entmax-attntokens-$tokens-attntokenlayers-$tokenlayers-data-ddsm-view-$view-$postfix"
                    run_train $experiment_id $fold \
                      --model $model \
                      --attention-heads $heads \
                      --attention-downsampling 1 \
                      --attention-l1-loss $l1loss \
                      --attention-combine $comb \
                      --attention-entmax-alpha $entmax \
                      --attention-tokens $tokens \
                      --attention-token-layers $tokenlayers \
                      --attention-implementation traditional \
                      --ddsm-views $view_list

                    experiment_id="$prefix-model-$model-attnheads-$heads-attndds-1-attncomb-$comb-attnl1-$l1loss-$entmax-attntokens-$tokens-attntokenlayers-$tokenlayers-attnbidirectional-data-ddsm-view-$view-$postfix"
                    run_train $experiment_id $fold \
                      --model $model \
                      --attention-heads $heads \
                      --attention-downsampling 1 \
                      --attention-l1-loss $l1loss \
                      --attention-combine $comb \
                      --attention-entmax-alpha $entmax \
                      --attention-tokens $tokens \
                      --attention-token-layers $tokenlayers \
                      --attention-implementation traditional \
                      --attention-bidirectional \
                      --ddsm-views $view_list
                  done
                done
              done
            done
          done
        done
      done

      for model in TwoViewAttentionResNet18ShallowTop ; do
        for comb in add add-linear ln-add-linear-do ; do
          for heads in 12 18 24 ; do
            experiment_id="$prefix-model-$model-attnheads-$heads-attndds-1-attncomb-$comb-attnl1-$l1loss-$entmax-data-ddsm-view-$view-$postfix"
            run_train $experiment_id $fold \
              --model $model \
              --attention-heads $heads \
              --attention-downsampling 1 \
              --attention-l1-loss $l1loss \
              --attention-combine $comb \
              --attention-entmax-alpha $entmax \
              --attention-implementation traditional \
              --ddsm-views $view_list

            experiment_id="$prefix-model-$model-attnheads-$heads-attndds-1-attncomb-$comb-attnl1-$l1loss-$entmax-attnbidirectional-data-ddsm-view-$view-$postfix"
            run_train $experiment_id $fold \
              --model $model \
              --attention-heads $heads \
              --attention-downsampling 1 \
              --attention-l1-loss $l1loss \
              --attention-combine $comb \
              --attention-entmax-alpha $entmax \
              --attention-implementation traditional \
              --attention-bidirectional \
              --ddsm-views $view_list
          done
        done
      done
    done
  done
done

echo
echo "#SBATCH --array=1-$counter"
echo
