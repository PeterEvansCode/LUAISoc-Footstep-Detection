#include <cstdio>
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"  // xxd output

// --- TFLM setup ---
constexpr int kTensorArenaSize = 128 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// --- Audio settings ---
constexpr int kAudioSampleRate = 16000; // Hz
constexpr int kAudioFrameSize = 160;    // samples per inference (~10ms at 16kHz)
int16_t audio_buffer[kAudioFrameSize];

// --- Footstep callback ---
void on_footstep_detected() {
    printf("Footstep detected!\n");
}

// --- Simple mock microphone read ---
void read_microphone(int16_t* buffer, int size) {
    // Replace with real ADC/PDM read
    for (int i = 0; i < size; i++) {
        buffer[i] = adc_read();  // 12-bit ADC, centered around 0
    }
}

// --- Convert ADC samples to float normalized for model ---
void preprocess_audio(int16_t* raw, float* processed, int size) {
    for (int i = 0; i < size; i++) {
        processed[i] = raw[i] / 2048.0f; // normalize int16 -> float [-1,1]
    }
}

int main() {
    stdio_init_all();
    sleep_ms(500);

    printf("Starting footstep detector...\n");

    // --- Initialize ADC (assuming microphone on ADC0) ---
    adc_init();
    adc_gpio_init(26); // GP26 = ADC0
    adc_select_input(0);

    // --- Load model ---
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        while (1);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed\n");
        while (1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    while (1) {
        // 1) Read audio frame
        read_microphone(audio_buffer, kAudioFrameSize);

        // 2) Preprocess
        for (int i = 0; i < kAudioFrameSize; i++) {
            input->data.f[i] = audio_buffer[i] / 2048.0f;
        }

        // 3) Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Inference failed!\n");
            continue;
        }

        // 4) Read output (assuming single float: probability of footstep)
        float prob = output->data.f[0];

        if (prob > 0.5f) {  // threshold for footstep
            on_footstep_detected();
        }

        sleep_ms(10); // adjust to match inference rate
    }
}
