"""Reference Flux implementation of the pneumonia kaggle challange."""

using Flux, CuArrays, Images, Glob, Random, Base.Iterators
using Lazy: @>
using Metalhead: VGG19

data_root = "chest_xray"
batch_size = 32
device = cpu

label_names = sort(readdir(joinpath(data_root, "train")))
label_to_index = Dict(name => index-1 for (index, name) in enumerate(label_names))

load_and_prepocess_image(path::String) =
    @> path begin
        load()
        imresize(224, 224)
        RGB.()
        channelview()
        permuteddimsview((2, 3, 1))
        rawview()
        collect()
    end


function dataset(path::String)
    image_paths = shuffle(glob("*/*", joinpath(data_root, path)))
    image_labels = [label_to_index[basename(dirname(path))]
                    for path in image_paths]
    images = @> begin
        (load_and_prepocess_image(path) for path in image_paths)
        partition(batch_size)
        batches -> (device(cat(batch..., dims=4)) for batch in batches)
    end

    labels = @> begin
        image_labels
        partition(batch_size)
        batches -> (device(batch) for batch in batches)
    end

    zip(images, labels)
end

train_dataset = dataset("train")

vgg = VGG19()

model = device(Chain(vgg, Dense(1000, 1, σ)))

optimizer = ADAM()

loss(x, y) = Flux.crossentropy(model(x), y)

Flux.train!(loss, params(model), train_dataset, optimizer)
