#!/bin/bash

datasets=("noise_image_lena_const.txt" "noise_image_tulips_const.txt" "noise_image_house_const.txt")

data_path="./data/"
datasets_prefixed="${datasets[@]/#/$data_path}"

mkdir -p results
echo "version,dataset,patch_size,blocks,threads,time" >>./results/results.csv

patch_sizes=(3 5 7)

run_v1() {

    block_sizes=(16 32 64 128 256 512 1024 2048 4094)
    thread_nums=(32 64 128 256 512 1024 2048)

    for dataset in ${datasets_prefixed[@]}; do
        printf "\n------------------ %25s -------------\n" "$dataset"
        for patch_size in ${patch_sizes[@]}; do
            for block_size in ${block_sizes[@]}; do
                for threads in ${thread_nums[@]}; do
                    export OMP_NUM_THREADS=$t
                    printf "%50s threads\n" "$t"
                    ./bin/v1 $dataset $patch_size $block_size $threads
                done
            done
        done
    done
}

run_v2() {

    block_sizes=(16 32 64 128 256 512 1024 2048 4094)
    thread_nums=(32 64 96 128 256 512 672)

    for dataset in ${datasets_prefixed[@]}; do
        printf "\n------------------ %25s -------------\n" "$dataset"
        for patch_size in ${patch_sizes[@]}; do
            for block_size in ${block_sizes[@]}; do
                for threads in ${thread_nums[@]}; do
                    export OMP_NUM_THREADS=$t
                    printf "%s patch_size | %s block_size | %s threads\n" "$patch_size" "$block_size" "$threads"
                    ./bin/v2 $dataset $patch_size $block_size $threads
                done
            done
        done
    done
}

run_v1
run_v2

# sweep_files_threads() {
#     for d in ${datasets_prefixed[@]}; do
#         printf "\n------------------ %25s -------------\n" "$d"
#         for t in ${thread_nums[@]}; do
#             export CILK_NWORKERS=$t
#             export OMP_NUM_THREADS=$t
#             printf "%50s threads\n" "$t"
#             ./build/main $d $t
#         done
#     done
# }

# run_library() {
#     for i in 1; do
#         echo "#define FRAMEWORK ${i}" >./include/framework.h
#         printf "\n==============================================================\n"
#         printf "\n[Framework] %s" "${make_options[$i]}"

#         if [ ${make_options[$i]} == "cilk" ]; then
#             module purge gcc
#             module load OpenCilk
#         else
#             module purge OpenCilk
#             module load gcc
#         fi

#         make ${make_options[$i]}

#         for i in $(seq 1 5); do
#             sweep_files_threads
#         done
#     done
# }

# for i in $(seq 1 5); do
#     sweep_files_threads
# done

# run_library
