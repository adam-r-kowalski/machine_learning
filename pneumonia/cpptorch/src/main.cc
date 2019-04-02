#include <torch/torch.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>

namespace fs = boost::filesystem;
namespace nn = torch::nn;
namespace data = torch::data;

struct AdaptiveAvgPool2d : nn::Module {
  auto forward(const torch::Tensor &x) -> torch::Tensor {
    return torch::adaptive_avg_pool2d(x, {1, 1}).squeeze_();
  }
};

const auto root_dir =
    fs::path("/Users/adamkowalski/src/machine_learning/pneumonia/chest_xray");

const auto label_to_index =
    std::map<fs::path, int>{{"NORMAL", 0}, {"PNEUMONIA", 1}};

const auto batch_size = 64;
const auto workers = std::thread::hardware_concurrency();
const auto phases = {"train", "val", "test"};
const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
const auto epochs = 3;

auto load_and_preprocess_image(const fs::path &path) -> torch::Tensor {
  auto image = cv::imread(path.string());
  cv::resize(image, image, cv::Size(195, 195));
  const auto tensor =
      torch::tensor(torch::ArrayRef<uint8_t>(
                        image.data, static_cast<size_t>(image.rows) *
                                        static_cast<size_t>(image.cols) * 3))
          .view({image.rows, image.cols, 3})
          .permute({2, 0, 1})
          .to(torch::kF32);
  return tensor / 127.5 - 1;
}

struct Pneumonia : data::Dataset<Pneumonia> {
  explicit Pneumonia(const fs::path &path) {
    auto rng = std::default_random_engine{};
    const auto normal_iter = fs::directory_iterator(path / "NORMAL");
    const auto pneumonia_iter = fs::directory_iterator(path / "PNEUMONIA");
    image_paths.insert(image_paths.end(), begin(normal_iter), end(normal_iter));
    image_paths.insert(image_paths.end(), begin(pneumonia_iter),
                       end(pneumonia_iter));
    std::shuffle(image_paths.begin(), image_paths.end(), rng);
  }

  auto get(size_t index) -> data::Example<> override {
    const auto label =
        label_to_index.at(image_paths[index].parent_path().filename());
    return {load_and_preprocess_image(image_paths[index]),
            torch::tensor(label).to(torch::kF32)};
  }

  auto size() const -> torch::optional<size_t> override {
    return image_paths.size();
  }

 private:
  std::vector<fs::path> image_paths;
};

template <class... Ts>
auto train(nn::Sequential &model, torch::optim::Optimizer &optimizer,
           data::DataLoaderBase<Ts...> &data_loader, size_t size) {
  model->train();

  std::cout << "training on dataset of size " << size << std::endl;

  auto running_loss = torch::tensor(0.0);
  auto running_corrects = torch::tensor(0);

  auto i = 1;
  auto batches = int(size / batch_size);

  for (auto &batch : data_loader) {
    std::cout << "batch " << i++ << " / " << batches << std::endl;

    const auto images = batch.data.to(device);
    const auto labels = batch.target.to(device);

    optimizer.zero_grad();
    const auto predictions = model->forward(images);
    auto loss = torch::binary_cross_entropy(predictions, labels);
    loss.backward();
    optimizer.step();

    running_loss += loss * images.size(0);
    running_corrects += (torch::round(predictions) == labels).sum();
  }

  std::cout << "\nepoch loss = " << running_loss.item<double>() / size
            << "\nepoch accuracy = " << running_corrects.item<double>() / size
            << std::endl;
}

template <class... Ts>
auto eval(nn::Sequential &model, data::DataLoaderBase<Ts...> &data_loader,
          size_t size) {
  model->eval();

  auto running_loss = torch::tensor(0.0);
  auto running_corrects = torch::tensor(0);

  std::cout << "evaluating dataset of size " << size << std::endl;

  for (auto &batch : data_loader) {
    const auto images = batch.data.to(device);
    const auto labels = batch.target.to(device);

    const auto predictions = model->forward(images);
    const auto loss = torch::binary_cross_entropy(predictions, labels);

    running_loss += loss * images.size(0);
    running_corrects += (torch::round(predictions) == labels).sum();
  }

  std::cout << "\nloss = " << running_loss.item<double>() / size
            << "\naccuracy = " << running_corrects.item<double>() / size
            << std::endl;
}

auto main() -> int {
  const auto train_dataset =
      Pneumonia{root_dir / "train"}.map(data::transforms::Stack());
  const auto train_loader = data::make_data_loader(
      train_dataset, data::DataLoaderOptions(batch_size).workers(workers));

  const auto val_dataset =
      Pneumonia{root_dir / "val"}.map(data::transforms::Stack());
  const auto val_loader = data::make_data_loader(
      val_dataset, data::DataLoaderOptions(batch_size).workers(workers));

  const auto test_dataset =
      Pneumonia{root_dir / "test"}.map(data::transforms::Stack());
  const auto test_loader = data::make_data_loader(
      test_dataset, data::DataLoaderOptions(batch_size).workers(workers));

  auto model = nn::Sequential(
      nn::Conv2d(nn::Conv2dOptions(3, 8, {3, 3})), nn::BatchNorm(8),
      nn::Functional(torch::relu), nn::Conv2d(nn::Conv2dOptions(8, 16, {3, 3})),
      nn::BatchNorm(16), nn::Functional(torch::relu),
      nn::Conv2d(nn::Conv2dOptions(16, 32, {3, 3})), nn::BatchNorm(32),
      nn::Functional(torch::relu), AdaptiveAvgPool2d(), nn::Linear(32, 1),
      nn::Functional(torch::sigmoid));
  model->to(device);

  auto optimizer = torch::optim::Adam(model->parameters(), /*lr=*/0.01);

  for (auto epoch = 0; epoch < epochs; ++epoch) {
    train(model, optimizer, *train_loader, *train_dataset.size());
    eval(model, *val_loader, *val_dataset.size());
  }

  eval(model, *test_loader, *test_dataset.size());

  return 0;
}
