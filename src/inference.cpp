#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Hàm chuẩn hóa ảnh
void normalize_image(cv::Mat& img, float mean, float std) {
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    img = (img - mean) / std;
}

// 1. Khởi tạo ONNX session
Ort::Session initialize_session(const std::string& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mnist_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    return Ort::Session(env, model_path.c_str(), session_options);
}

// 2. Đọc và tiền xử lý ảnh
std::vector<float> load_and_preprocess_image(const std::string& sample_path, size_t input_size) {
    std::vector<float> input_data(input_size);
    cv::Mat img = cv::imread(sample_path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Không thể đọc ảnh: " << sample_path << std::endl;
        std::fill(input_data.begin(), input_data.end(), 0.0f);
    } else {
        std::cout << "Đọc ảnh thành công: " << sample_path << std::endl;
        if (img.rows != 28 || img.cols != 28) {
            cv::resize(img, img, cv::Size(28, 28));
        }
        normalize_image(img, 0.1307f, 0.3081f);

        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                input_data[i * 28 + j] = img.at<float>(i, j);
            }
        }
    }
    return input_data;
}

// 3. Tạo tensor đầu vào
Ort::Value create_input_tensor(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_data.data()), input_data.size(), 
        input_shape.data(), input_shape.size()
    );
}

// 4. Chạy inference và đo thời gian
std::pair<std::vector<Ort::Value>, float> run_inference(
    Ort::Session& session, Ort::Value& input_tensor, 
    const char* input_names[], const char* output_names[]
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );
    auto end_time = std::chrono::high_resolution_clock::now();
    float inference_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    return {std::move(output_tensors), inference_time};
}

// 5. Xử lý đầu ra (sửa lỗi const)
int process_output(std::vector<Ort::Value>& output_tensors) {
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> output(output_data, output_data + 10);
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

// 6. Hiển thị kết quả
void display_result(int predicted, float inference_time, const cv::Mat& img, bool plot) {
    std::cout << "Predicted class: " << predicted << std::endl;
    std::cout << "Inference time: " << inference_time << " ms" << std::endl;
    std::cout << std::endl;

    if (!img.empty() && plot) {
        cv::imshow("Input Image", img);
        cv::waitKey(0);
    }
}

// Hàm chính
void infer(std::string model_path, std::string sample_path, bool plot = false) {
    std::vector<int64_t> input_shape = {1, 1, 28, 28};
    size_t input_size = 1 * 1 * 28 * 28;
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output"};

    Ort::Session session = initialize_session(model_path);
    std::vector<float> input_data = load_and_preprocess_image(sample_path, input_size);
    Ort::Value input_tensor = create_input_tensor(input_data, input_shape);
    auto [output_tensors, inference_time] = run_inference(session, input_tensor, input_names, output_names);
    int predicted = process_output(output_tensors);

    cv::Mat img = cv::imread(sample_path, cv::IMREAD_GRAYSCALE);
    if (!img.empty() && (img.rows != 28 || img.cols != 28)) {
        cv::resize(img, img, cv::Size(28, 28));
    }
    display_result(predicted, inference_time, img, plot);
}

int main() {
    std::string model_path = "weights/best_model.onnx";
    std::string sample_path_base = "sample/sample_digit";

    for (int i = 1; i < 10; ++i) {
        std::string sample_path = sample_path_base + std::to_string(i) + ".png";
        std::cout << sample_path << std::endl;
        infer(model_path, sample_path);
    }

    return 0;
}