#include <torch/torch.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = boost::filesystem;
namespace nn = torch::nn;

struct AdaptiveAvgPool2d : nn::Module {
  auto forward(const torch::Tensor &x) -> torch::Tensor {
    return torch::adaptive_avg_pool2d(x, {1, 1}).squeeze_();
  }
};

auto load_and_preprocess_image(const std::string &path) -> torch::Tensor {
  auto image = cv::imread(path);
  cv::resize(image, image, cv::Size(195, 195));
  auto tensor =
      torch::tensor(torch::ArrayRef<uint8_t>(
                        image.data, static_cast<size_t>(image.rows) *
                                        static_cast<size_t>(image.cols) * 3))
          .view({1, image.rows, image.cols, 3})
          .permute({0, 3, 1, 2})
          .to(torch::kF32);
  return tensor / 127.5 - 1;
}

auto main() -> int {
  auto model = nn::Sequential(
      nn::Conv2d(nn::Conv2dOptions(3, 8, {3, 3})), nn::BatchNorm(8),
      nn::Functional(torch::relu), nn::Conv2d(nn::Conv2dOptions(8, 16, {3, 3})),
      nn::BatchNorm(16), nn::Functional(torch::relu),
      nn::Conv2d(nn::Conv2dOptions(16, 32, {3, 3})), nn::BatchNorm(32),
      nn::Functional(torch::relu), AdaptiveAvgPool2d(), nn::Linear(32, 1),
      nn::Functional(torch::sigmoid));

  const auto root_dir = fs::path(
      "/Users/adamkowalski/src/machine_learning/pneumonia/chest_xray/train/"
      "NORMAL");

  const auto image_path =
      begin(fs::directory_iterator(root_dir))->path().string();

  const auto image = load_and_preprocess_image(image_path);
  std::cout << "image size = " << image.sizes() << "\nmin = " << image.min()
            << "\nmax = " << image.max() << "\nmean = " << image.mean()
            << "\nstd = " << image.std();

  const auto prediction = model->forward(image);
  std::cout << "prediction size = " << prediction.sizes() << "\n";

  return 0;
}
