#include <iostream>
#include <sndfile.h>
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <png.h> 

// Hàm lưu spectrogram thành file PNG
void saveSpectrogramToPNG(const std::vector<std::vector<double>>& spectrogram, const char* filename) {
    int height = spectrogram.size();         // Số frame (trục thời gian)
    int width = spectrogram[0].size();       // Số bin tần số

    // Tìm giá trị lớn nhất để chuẩn hóa
    double max_val = 0.0;
    for (const auto& row : spectrogram) {
        for (double val : row) {
            if (val > max_val) max_val = val;
        }
    }

    // Mở file PNG để ghi
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "Không thể tạo file PNG!" << std::endl;
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Chuẩn bị dữ liệu ảnh
    std::vector<png_byte> row_data(width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Chuẩn hóa giá trị thành 0-255
            double normalized = (spectrogram[y][x] / max_val) * 255.0;
            row_data[x] = static_cast<png_byte>(normalized);
        }
        png_write_row(png, row_data.data());
    }

    // Hoàn tất và đóng file
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

int main() {
    // Mở file WAV
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open("data/audio/audio.flac", SFM_READ, &sfinfo);
    
    if (!infile) {
        std::cerr << "Không thể mở file!" << std::endl;
        return 1;
    }

    // Đọc dữ liệu âm thanh
    std::vector<double> audio_data(sfinfo.frames * sfinfo.channels);
    sf_read_double(infile, audio_data.data(), sfinfo.frames * sfinfo.channels);
    sf_close(infile);

    // Thiết lập thông số FFT
    const int window_size = 1024;  // Kích thước cửa sổ FFT
    const int hop_size = 256;      // Độ dịch chuyển giữa các frame
    
    // Tính số lượng frame
    int num_frames = (audio_data.size() - window_size) / hop_size + 1;
    
    // Chuẩn bị input/output cho FFTW
    double* in = (double*) fftw_malloc(sizeof(double) * window_size);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (window_size/2 + 1));
    //fftw_plan_dft_r2c_1d: real-to-complex (thực sang phức) trong một chiều (1D) 
    fftw_plan plan = fftw_plan_dft_r2c_1d(window_size, in, out, FFTW_ESTIMATE);

    // Tạo ma trận spectrogram
    std::vector<std::vector<double>> spectrogram(num_frames, std::vector<double>(window_size/2 + 1));

    // Hàm cửa sổ Hanning
    std::vector<double> window(window_size);
    for (int i = 0; i < window_size; i++) {
        window[i] = 0.5 * (1 - cos(2 * M_PI * i / (window_size - 1)));
    }

    // Tính spectrogram
    for (int frame = 0; frame < num_frames; frame++) {
        // Chuẩn bị dữ liệu input
        for (int i = 0; i < window_size; i++) {
            int idx = frame * hop_size + i;
            in[i] = (idx < audio_data.size()) ? audio_data[idx] * window[i] : 0.0;
        }

        // Thực hiện FFT
        fftw_execute(plan);

        // Tính magnitude spectrum
        for (int i = 0; i < window_size/2 + 1; i++) {
            double real = out[i][0];
            double imag = out[i][1];
            spectrogram[frame][i] = sqrt(real*real + imag*imag);
        }
    }

    // Lưu spectrogram thành file PNG thay vì in ra màn hình
    saveSpectrogramToPNG(spectrogram, "out/spectrogram.png");
    std::cout << "Đã lưu spectrogram vào file 'spectrogram.png'" << std::endl;

    // Giải phóng bộ nhớ
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    fftw_cleanup();

    return 0;
}