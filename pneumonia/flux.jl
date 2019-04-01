"""Reference Flux implementation of the pneumonia kaggle challange."""

using Distributed, Base.Threads
addprocs(nthreads())

@everywhere using Images
@everywhere using Lazy: @>
using Flux, CuArrays, Glob, Random, Base.Iterators
using Metalhead: VGG19

cd("machine_learning/pneumonia/")


const data_root = "chest_xray"
const batch_size = 32
const device = gpu

const label_names = sort(readdir(joinpath(data_root, "train")))
const label_to_index = Dict(name => index-1 for (index, name) in enumerate(label_names))

@everywhere load_and_preprocess_image(path::String) =
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
        pmap(load_and_preprocess_image, image_paths)
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

function eval_model(model::Chain, dataset)
    running_loss = 0.0
    running_corrects = 0
    for (inputs, labels) ∈ dataset
        preds = model(inputs)
        loss = Flux.crossentropy(preds, labels)
        running_loss += loss * length(labels)
        running_corrects += sum(round.(preds) .== labels')
    end
    examples = length(dataset) * batch_size
    epoch_loss = running_loss / examples
    epoch_acc = running_corrects / examples
    epoch_loss, epoch_acc
end

train_dataset = dataset("train")
val_dataset = dataset("val")
test_dataset = dataset("test")

vgg = VGG19()

model = Chain(
    vgg,
    Dense(1000, 100),
    BatchNorm(100),
    relu,
    Dropout(0.5),
    Dense(100, 1, σ)
    ) |> device

optimizer = ADAM()

loss(x, y) = Flux.crossentropy(model(x), y)

Flux.train!(loss, params(model), train_dataset, optimizer)
eval_model(model, train_dataset)
eval_model(model, val_dataset)
eval_model(model, test_dataset)
