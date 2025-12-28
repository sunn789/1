# Speech Signal Processing Project

[English](#english) | [فارسی](#فارسی)

---

## English

### Project Overview

This project focuses on speech signal processing in a noisy industrial environment. The goal is to understand speech signal behavior in time and frequency domains, calculate primary features, investigate different types of noise, and reduce their impact on speech signals.

### Problem Statement

In a noisy industrial environment, radio communication between technicians is disrupted. This project analyzes and processes raw speech signals to:
- Understand speech signal behavior in time and frequency domains
- Calculate primary signal features
- Investigate different types of noise
- Reduce noise impact on speech signals

### Project Structure

```
.
├── part1_signal_loading.py          # Part 1: Signal loading and display
├── part2_frequency_analysis.py       # Part 2: Frequency domain analysis
├── part3_noise_modeling.py          # Part 3: Noise identification and modeling
├── part4_noise_removal.py           # Part 4: Noise removal and signal reconstruction
├── speech_processing.py              # Complete integrated script
├── sound1.wav                        # Input audio file
├── output_images/                    # Generated plots and figures
├── ReadMe.md                         # Detailed documentation (Persian)
└── README.md                         # This file
```

### Requirements

```bash
pip install numpy scipy matplotlib tabulate
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd speech-processing-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have `sound1.wav` in the project root directory.

### Usage

#### Running Individual Parts

**Part 1: Signal Loading and Display**
```bash
python part1_signal_loading.py
```
- Loads and displays the speech signal
- Calculates signal properties (length, sampling rate, number of samples)
- Explains the importance of sampling rate in speech processing

**Part 2: Frequency Analysis**
```bash
python part2_frequency_analysis.py
```
- Calculates frequency spectrum using FFT
- Plots frequency spectrum
- Determines frequency ranges with maximum speech energy

**Part 3: Noise Modeling**
```bash
python part3_noise_modeling.py
```
- Adds Gaussian noise, Babble noise, and Noisex-92 noise to the signal
- Calculates SNR before and after adding noise

**Part 4: Noise Removal**
```bash
python part4_noise_removal.py
```
- Designs FIR and IIR filters
- Applies filters to noisy signals
- Compares filter performance

#### Running Complete Script

```bash
python speech_processing.py
```
Runs all parts sequentially and generates comprehensive results.

### Project Parts

#### Part 1: Reading and Displaying Speech Signal

**1a. Load and Display Signal**
- Loads audio file using `scipy.io.wavfile`
- Displays signal in time domain
- Converts stereo to mono if needed
- Normalizes signal amplitude

**1b. Calculate Signal Properties**
- Signal length (seconds)
- Sampling rate (Hz)
- Number of samples
- Statistical properties (mean, std, max, min)

**1c. Importance of Sampling Rate**
- According to Nyquist-Shannon theorem, sampling rate must be at least twice the highest frequency
- Standard sampling rates: 8 kHz (telephone), 16 kHz (speech), 44.1 kHz (audio)
- Prevents aliasing phenomenon
- Affects data volume, bandwidth, and computational complexity

#### Part 2: Frequency Analysis of Speech Signal

**2a. Calculate Frequency Spectrum using FFT**
- Computes FFT of the signal
- Calculates magnitude spectrum
- Determines frequency resolution

**2b. Plot Frequency Spectrum and Energy Distribution**
- Plots full frequency spectrum
- Plots speech range (0-4000 Hz)
- Calculates energy distribution in frequency bands:
  - 0-300 Hz: Pitch and emotional information
  - 300-1000 Hz: Formant F1
  - 1000-3000 Hz: Formants F2 and F3 (maximum speech energy)
  - 3000-8000 Hz: Unvoiced sounds and fine details

**Results:**
- Maximum speech energy typically in 300-3000 Hz range
- Fundamental frequency: 85-255 Hz (men), 165-255 Hz (women)
- Formants: F1 (200-800 Hz), F2 (800-2500 Hz), F3 (2500-3500 Hz)

#### Part 3: Noise Identification and Modeling

**3a. Add Different Types of Noise**
- **Gaussian Noise**: White noise with normal distribution, uniform spectral distribution
- **Babble Noise**: Simulated by combining multiple sinusoidal signals, mimics crowded environments
- **Noisex-92 Noise**: Industrial noise combining Gaussian and impulsive components

**3b. Calculate SNR**
- SNR (dB) = 10 × log₁₀(signal_power / noise_power)
- Calculates SNR before and after adding noise
- Target SNR: 10 dB (simulating noisy industrial environment)

**SNR Levels:**
- > 30 dB: Excellent quality (studio)
- 20-30 dB: Good quality (quiet environment)
- 10-20 dB: Moderate quality (office)
- < 10 dB: Poor quality (noisy street)

#### Part 4: Noise Removal and Signal Reconstruction

**4a. Design FIR and IIR Filters**

**FIR Filters:**
- Finite Impulse Response
- Linear phase (no temporal distortion)
- Always stable
- Higher computational complexity
- Designed using Hamming window

**IIR Filters:**
- Infinite Impulse Response
- More efficient at lower orders
- Sharper frequency response
- May be unstable
- Non-linear phase (temporal distortion)
- Types implemented:
  - Butterworth
  - Chebyshev Type 1
  - Chebyshev Type 2
  - Elliptic

**Filter Parameters:**
- Passband: 80-3800 Hz (speech frequency range)
- FIR order: 101 taps
- IIR order: 4

**4b. Filter Performance Comparison**

**Results:**
- For Gaussian noise: IIR filters (especially Chebyshev Type 1) perform better
- For Babble noise: FIR filter performs better due to linear phase
- For Noisex-92 noise: IIR Elliptic and Chebyshev Type 2 perform best

### Output Files

#### Images (in `output_images/` directory)
- `speech_signal_time_domain.png` - Signal in time domain
- `speech_signal_frequency_domain.png` - Frequency spectrum
- `part2_frequency_spectrum.png` - Detailed frequency analysis
- `part2_energy_distribution.png` - Energy distribution in frequency bands
- `noisy_signals_comparison.png` - Comparison of noisy signals
- `filters_frequency_response.png` - Filter frequency responses
- `filters_performance_comparison.png` - Filter performance comparison

#### Text Files
- `signal_info_table.txt` - Signal properties
- `energy_distribution_table.txt` - Energy distribution
- `snr_comparison_table.txt` - SNR comparison
- `part2_frequency_analysis_results.txt` - Frequency analysis results
- `part3_noise_analysis_results.txt` - Noise analysis results
- `part4_performance_*.txt` - Filter performance for each noise type
- `part4_comparison_summary.txt` - Overall filter comparison

### Key Findings

1. **Sampling Rate Importance**: Critical for preventing aliasing and maintaining signal quality
2. **Energy Distribution**: Maximum speech energy in 300-3000 Hz range
3. **Noise Types**: Different noise types require different filtering approaches
4. **Filter Selection**: 
   - FIR for preserving speech quality (linear phase)
   - IIR for better noise reduction (higher efficiency)
5. **SNR Improvement**: Filters can improve SNR by 2-5 dB depending on noise type

### Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computations
- **SciPy**: Signal processing and FFT
- **Matplotlib**: Visualization
- **Tabulate**: Table formatting

### Author

Speech Processing Course - Assignment 1

### License

This project is for educational purposes.

---

## فارسی

### نمای کلی پروژه

این پروژه بر روی پردازش سیگنال گفتار در یک محیط صنعتی پرسر و صدا تمرکز دارد. هدف درک رفتار سیگنال گفتار در حوزه زمان و فرکانس، محاسبه ویژگی‌های اولیه، بررسی انواع مختلف نویز و کاهش تأثیر آن‌ها بر سیگنال‌های گفتار است.

### بیان مسئله

در یک محیط صنعتی پرسر و صدا، ارتباط رادیویی بین تکنسین‌ها مختل شده است. این پروژه سیگنال‌های گفتار خام را تحلیل و پردازش می‌کند تا:
- رفتار سیگنال گفتار را در حوزه زمان و فرکانس درک کند
- ویژگی‌های اولیه سیگنال را محاسبه کند
- انواع مختلف نویز را بررسی کند
- تأثیر نویز بر سیگنال‌های گفتار را کاهش دهد

### ساختار پروژه

```
.
├── part1_signal_loading.py          # بخش ۱: خواندن و نمایش سیگنال
├── part2_frequency_analysis.py       # بخش ۲: تحلیل فرکانسی
├── part3_noise_modeling.py          # بخش ۳: شناسایی و مدلسازی نویز
├── part4_noise_removal.py           # بخش ۴: حذف نویز و بازسازی
├── speech_processing.py              # اسکریپت کامل یکپارچه
├── sound1.wav                        # فایل صوتی ورودی
├── output_images/                    # نمودارها و تصاویر تولید شده
├── ReadMe.md                         # مستندات تفصیلی (فارسی)
└── README.md                         # این فایل
```

### نیازمندی‌ها

```bash
pip install numpy scipy matplotlib tabulate
```

### نصب

1. کلون کردن مخزن:
```bash
git clone <repository-url>
cd speech-processing-project
```

2. نصب وابستگی‌ها:
```bash
pip install -r requirements.txt
```

3. اطمینان حاصل کنید که `sound1.wav` در دایرکتوری ریشه پروژه وجود دارد.

### نحوه استفاده

#### اجرای بخش‌های جداگانه

**بخش ۱: خواندن و نمایش سیگنال**
```bash
python part1_signal_loading.py
```
- سیگنال گفتار را بارگذاری و نمایش می‌دهد
- مشخصات سیگنال را محاسبه می‌کند (طول، نرخ نمونه‌برداری، تعداد نمونه‌ها)
- اهمیت نرخ نمونه‌برداری در پردازش گفتار را توضیح می‌دهد

**بخش ۲: تحلیل فرکانسی**
```bash
python part2_frequency_analysis.py
```
- طیف فرکانسی را با استفاده از FFT محاسبه می‌کند
- نمودار طیف فرکانسی را رسم می‌کند
- بازه‌های فرکانسی با بیشترین انرژی گفتار را تعیین می‌کند

**بخش ۳: مدلسازی نویز**
```bash
python part3_noise_modeling.py
```
- نویز گوسی، نویز همهمه و نویز Noisex-92 را به سیگنال اضافه می‌کند
- SNR را قبل و بعد از اضافه کردن نویز محاسبه می‌کند

**بخش ۴: حذف نویز**
```bash
python part4_noise_removal.py
```
- فیلترهای FIR و IIR را طراحی می‌کند
- فیلترها را بر روی سیگنال‌های نویزی اعمال می‌کند
- عملکرد فیلترها را مقایسه می‌کند

#### اجرای اسکریپت کامل

```bash
python speech_processing.py
```
همه بخش‌ها را به ترتیب اجرا می‌کند و نتایج جامعی تولید می‌کند.

### بخش‌های پروژه

#### بخش ۱: خواندن و نمایش سیگنال گفتار

**۱الف. بارگذاری و نمایش سیگنال**
- فایل صوتی را با استفاده از `scipy.io.wavfile` بارگذاری می‌کند
- سیگنال را در حوزه زمان نمایش می‌دهد
- در صورت نیاز استریو را به مونو تبدیل می‌کند
- دامنه سیگنال را نرمال‌سازی می‌کند

**۱ب. محاسبه مشخصات سیگنال**
- طول سیگنال (ثانیه)
- نرخ نمونه‌برداری (هرتز)
- تعداد نمونه‌ها
- ویژگی‌های آماری (میانگین، انحراف معیار، حداکثر، حداقل)

**۱ج. اهمیت نرخ نمونه‌برداری**
- طبق قضیه نایکویست-شانون، نرخ نمونه‌برداری باید حداقل دو برابر بالاترین فرکانس باشد
- نرخ‌های نمونه‌برداری استاندارد: ۸ کیلوهرتز (تلفن)، ۱۶ کیلوهرتز (گفتار)، ۴۴.۱ کیلوهرتز (صوت)
- از پدیده علیاس جلوگیری می‌کند
- بر حجم داده‌ها، پهنای باند و پیچیدگی محاسباتی تأثیر می‌گذارد

#### بخش ۲: تحلیل فرکانسی سیگنال گفتار

**۲الف. محاسبه طیف فرکانسی با استفاده از FFT**
- FFT سیگنال را محاسبه می‌کند
- طیف دامنه را محاسبه می‌کند
- رزولوشن فرکانسی را تعیین می‌کند

**۲ب. رسم طیف فرکانسی و توزیع انرژی**
- طیف فرکانسی کامل را رسم می‌کند
- محدوده گفتار (۰-۴۰۰۰ هرتز) را رسم می‌کند
- توزیع انرژی در باندهای فرکانسی را محاسبه می‌کند:
  - ۰-۳۰۰ هرتز: اطلاعات زیروبمی و احساسی
  - ۳۰۰-۱۰۰۰ هرتز: فرمانت F1
  - ۱۰۰۰-۳۰۰۰ هرتز: فرمانت‌های F2 و F3 (حداکثر انرژی گفتار)
  - ۳۰۰۰-۸۰۰۰ هرتز: اصوات بی‌واک و جزئیات ظریف

**نتایج:**
- بیشترین انرژی گفتار معمولاً در بازه ۳۰۰-۳۰۰۰ هرتز قرار دارد
- فرکانس پایه: ۸۵-۲۵۵ هرتز (مردان)، ۱۶۵-۲۵۵ هرتز (زنان)
- فرمانت‌ها: F1 (۲۰۰-۸۰۰ هرتز)، F2 (۸۰۰-۲۵۰۰ هرتز)، F3 (۲۵۰۰-۳۵۰۰ هرتز)

#### بخش ۳: شناسایی و مدلسازی نویز

**۳الف. اضافه کردن انواع مختلف نویز**
- **نویز گوسی**: نویز سفید با توزیع نرمال، توزیع طیفی یکنواخت
- **نویز همهمه**: با ترکیب چندین سیگنال سینوسی شبیه‌سازی شده، محیط‌های شلوغ را شبیه‌سازی می‌کند
- **نویز Noisex-92**: نویز صنعتی که ترکیبی از مولفه‌های گوسی و ضربه‌ای است

**۳ب. محاسبه SNR**
- SNR (dB) = ۱۰ × log₁₀(توان_سیگنال / توان_نویز)
- SNR را قبل و بعد از اضافه کردن نویز محاسبه می‌کند
- SNR هدف: ۱۰ دسی‌بل (شبیه‌سازی محیط صنعتی پرسر و صدا)

**سطوح SNR:**
- > ۳۰ دسی‌بل: کیفیت عالی (استودیو)
- ۲۰-۳۰ دسی‌بل: کیفیت خوب (محیط آرام)
- ۱۰-۲۰ دسی‌بل: کیفیت متوسط (دفتر کار)
- < ۱۰ دسی‌بل: کیفیت ضعیف (خیابان شلوغ)

#### بخش ۴: حذف نویز و بازسازی سیگنال

**۴الف. طراحی فیلترهای FIR و IIR**

**فیلترهای FIR:**
- پاسخ ضربه‌ای محدود
- فاز خطی (بدون اعوجاج زمانی)
- همیشه پایدار
- پیچیدگی محاسباتی بالاتر
- با استفاده از پنجره Hamming طراحی شده است

**فیلترهای IIR:**
- پاسخ ضربه‌ای نامحدود
- کارایی بالاتر در مرتبه‌های پایین
- پاسخ فرکانسی تیزتر
- ممکن است ناپایدار باشد
- فاز غیرخطی (اعوجاج زمانی)
- انواع پیاده‌سازی شده:
  - باترورث
  - چبیشف نوع ۱
  - چبیشف نوع ۲
  - بیضوی

**پارامترهای فیلتر:**
- باند عبور: ۸۰-۳۸۰۰ هرتز (محدوده فرکانسی گفتار)
- مرتبه FIR: ۱۰۱ ضریب
- مرتبه IIR: ۴

**۴ب. مقایسه عملکرد فیلترها**

**نتایج:**
- برای نویز گوسی: فیلترهای IIR (به ویژه چبیشف نوع ۱) عملکرد بهتری دارند
- برای نویز همهمه: فیلتر FIR به دلیل فاز خطی عملکرد بهتری دارد
- برای نویز Noisex-92: فیلترهای IIR بیضوی و چبیشف نوع ۲ بهترین عملکرد را دارند

### فایل‌های خروجی

#### تصاویر (در پوشه `output_images/`)
- `speech_signal_time_domain.png` - سیگنال در حوزه زمان
- `speech_signal_frequency_domain.png` - طیف فرکانسی
- `part2_frequency_spectrum.png` - تحلیل تفصیلی فرکانسی
- `part2_energy_distribution.png` - توزیع انرژی در باندهای فرکانسی
- `noisy_signals_comparison.png` - مقایسه سیگنال‌های نویزی
- `filters_frequency_response.png` - پاسخ فرکانسی فیلترها
- `filters_performance_comparison.png` - مقایسه عملکرد فیلترها

#### فایل‌های متنی
- `signal_info_table.txt` - مشخصات سیگنال
- `energy_distribution_table.txt` - توزیع انرژی
- `snr_comparison_table.txt` - مقایسه SNR
- `part2_frequency_analysis_results.txt` - نتایج تحلیل فرکانسی
- `part3_noise_analysis_results.txt` - نتایج تحلیل نویز
- `part4_performance_*.txt` - عملکرد فیلتر برای هر نوع نویز
- `part4_comparison_summary.txt` - مقایسه کلی فیلترها

### یافته‌های کلیدی

1. **اهمیت نرخ نمونه‌برداری**: برای جلوگیری از علیاس و حفظ کیفیت سیگنال حیاتی است
2. **توزیع انرژی**: بیشترین انرژی گفتار در بازه ۳۰۰-۳۰۰۰ هرتز
3. **انواع نویز**: انواع مختلف نویز نیاز به رویکردهای فیلتر کردن متفاوت دارند
4. **انتخاب فیلتر**: 
   - FIR برای حفظ کیفیت گفتار (فاز خطی)
   - IIR برای کاهش بهتر نویز (کارایی بالاتر)
5. **بهبود SNR**: فیلترها می‌توانند SNR را بسته به نوع نویز ۲-۵ دسی‌بل بهبود دهند

### فناوری‌های استفاده شده

- **Python 3.x**
- **NumPy**: محاسبات عددی
- **SciPy**: پردازش سیگنال و FFT
- **Matplotlib**: تجسم داده‌ها
- **Tabulate**: فرمت‌بندی جداول

### نویسنده

درس پردازش گفتار - تمرین ۱

### مجوز

این پروژه برای اهداف آموزشی است.

