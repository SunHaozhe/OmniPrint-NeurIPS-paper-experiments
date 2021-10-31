
WORKDIR="../omniprint"
OUTPUTDIR="../transfer_mnist/tmp_dataset"

(cd ${WORKDIR} && python3 run.py --dict alphabets/fine/ascii_digits --count 3000 --equal_char --output_dir ${OUTPUTDIR} --output_subdir "mnist_like" --ensure_square_layout --gaussian_prior_resizing 4 --margins "0.1,0.1,0.1,0.1" --random_translation_x --random_translation_y --size 28 --background "0,0,0" --stroke_fill "255,255,255" --pre_elastic 0.03 --rotation -30 30 --shear_x -0.3 0.3)

(cd "${WORKDIR}/dataset" && python3 torch_image_dataset_formatter.py --dataset_name MNIST_like --raw_dataset_path ../../transfer_mnist/mnist_like --output_dir MNIST_like_folder)

(cd ${WORKDIR}/dataset && mv MNIST_like_folder/MNIST_like ../../transfer_mnist/MNIST_like_)

python3 rename_class_folders.py

