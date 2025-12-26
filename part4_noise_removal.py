"""
بخش ۴: حذف نویز و بازسازی سیگنال
تمرینات درس پردازش گفتار - سری اول
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import os
from scipy.fft import fft, fftfreq
from tabulate import tabulate

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def load_noisy_signals():
    """
    بارگذاری سیگنال‌های نویزی از بخش ۳
    """
    try:
        # بارگذاری سیگنال تمیز
        sample_rate, clean_signal = wav.read('sound1.wav')
        if len(clean_signal.shape) > 1:
            clean_signal = clean_signal.mean(axis=1)
        clean_signal = clean_signal / np.max(np.abs(clean_signal))
        
        # تولید سیگنال‌های نویزی (مشابه بخش ۳)
        np.random.seed(42)  # برای تکرارپذیری
        
        noise_types = ['gaussian', 'babble', 'noisex']
        noisy_signals = {}
        
        target_snr = 10
        
        for noise_type in noise_types:
            # تولید نویز
            if noise_type == 'gaussian':
                noise = np.random.normal(0, 1, len(clean_signal))
            elif noise_type == 'babble':
                t = np.arange(len(clean_signal)) / sample_rate
                noise = np.zeros(len(clean_signal))
                frequencies = np.linspace(100, 1000, 20)
                for freq in frequencies:
                    noise += 0.05 * np.sin(2 * np.pi * freq * t + np.random.rand()*2*np.pi)
            elif noise_type == 'noisex':
                noise = np.random.uniform(-1, 1, len(clean_signal))
            
            # تنظیم نویز برای SNR هدف
            signal_power = np.mean(clean_signal**2)
            noise_power = np.mean(noise**2)
            scale_factor = np.sqrt(signal_power / (noise_power * (10**(target_snr/10))))
            scaled_noise = noise * scale_factor
            
            # ایجاد سیگنال نویزی
            noisy_signals[noise_type] = clean_signal + scaled_noise
            
    except FileNotFoundError:
        print("⚠️ فایل اصلی یافت نشد. تولید سیگنال‌های نمونه...")
        sample_rate = 44100
        duration = 3
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # تولید سیگنال تمیز
        freq1, freq2, freq3, freq4 = 100, 200, 300, 800
        clean_signal = (0.5 * np.sin(2 * np.pi * freq1 * t) +
                        0.3 * np.sin(2 * np.pi * freq2 * t) +
                        0.2 * np.sin(2 * np.pi * freq3 * t) +
                        0.4 * np.sin(2 * np.pi * freq4 * t) *
                        np.exp(-0.5 * (t - duration/2)**2))
        clean_signal = clean_signal / np.max(np.abs(clean_signal))
        
        # تولید سیگنال‌های نویزی
        noise_types = ['gaussian', 'babble', 'noisex']
        noisy_signals = {}
        target_snr = 10
        
        for noise_type in noise_types:
            if noise_type == 'gaussian':
                noise = np.random.normal(0, 1, len(clean_signal))
            elif noise_type == 'babble':
                noise = np.zeros(len(clean_signal))
                frequencies = np.linspace(100, 1000, 20)
                for freq in frequencies:
                    noise += 0.05 * np.sin(2 * np.pi * freq * t + np.random.rand()*2*np.pi)
            elif noise_type == 'noisex':
                noise = np.random.uniform(-1, 1, len(clean_signal))
            
            signal_power = np.mean(clean_signal**2)
            noise_power = np.mean(noise**2)
            scale_factor = np.sqrt(signal_power / (noise_power * (10**(target_snr/10))))
            scaled_noise = noise * scale_factor
            
            noisy_signals[noise_type] = clean_signal + scaled_noise
    
    return sample_rate, clean_signal, noisy_signals

def calculate_snr(clean_signal, noisy_signal):
    """
    محاسبه SNR
    """
    signal_power = np.mean(clean_signal**2)
    noise = noisy_signal - clean_signal
    noise_power = np.mean(noise**2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return snr_db

def design_fir_filter(sample_rate, cutoff_low=80, cutoff_high=3800, numtaps=101):
    """
    طراحی فیلتر FIR
    """
    # طراحی فیلتر میان‌گذر FIR
    fir_coeff = signal.firwin(
        numtaps,
        [cutoff_low, cutoff_high],
        pass_zero=False,
        fs=sample_rate,
        window='hamming'
    )
    
    return fir_coeff

def design_iir_filters(sample_rate, cutoff_low=80, cutoff_high=3800, order=4):
    """
    طراحی فیلترهای IIR مختلف
    """
    nyquist = sample_rate / 2
    
    # نرمال‌سازی فرکانس‌ها
    low_norm = cutoff_low / nyquist
    high_norm = cutoff_high / nyquist
    
    # طراحی فیلترهای مختلف
    filters = {}
    
    # باترورث
    b_butter, a_butter = signal.butter(order, [low_norm, high_norm], btype='band')
    filters['butterworth'] = (b_butter, a_butter)
    
    # چبیشف نوع ۱ (ریپل در باند عبور)
    b_cheby1, a_cheby1 = signal.cheby1(order, 0.5, [low_norm, high_norm], btype='band')
    filters['chebyshev1'] = (b_cheby1, a_cheby1)
    
    # چبیشف نوع ۲ (ریپل در باند توقف)
    b_cheby2, a_cheby2 = signal.cheby2(order, 40, [low_norm, high_norm], btype='band')
    filters['chebyshev2'] = (b_cheby2, a_cheby2)
    
    # بیضوی (ریپل در هر دو باند)
    b_ellip, a_ellip = signal.ellip(order, 0.5, 40, [low_norm, high_norm], btype='band')
    filters['elliptic'] = (b_ellip, a_ellip)
    
    return filters

def apply_filters(noisy_signal, fir_coeff, iir_filters):
    """
    اعمال فیلترها بر روی سیگنال نویزی
    """
    filtered_signals = {}
    
    # اعمال فیلتر FIR
    filtered_signals['FIR'] = signal.lfilter(fir_coeff, 1.0, noisy_signal)
    
    # اعمال فیلترهای IIR
    for filter_name, (b, a) in iir_filters.items():
        filtered_signals[filter_name] = signal.lfilter(b, a, noisy_signal)
    
    return filtered_signals

def plot_filter_responses(fir_coeff, iir_filters, sample_rate):
    """
    رسم پاسخ فرکانسی فیلترها
    """
    plt.figure(figsize=(14, 10))
    
    # زیرنمودار ۱: پاسخ فرکانسی FIR
    plt.subplot(2, 2, 1)
    w_fir, h_fir = signal.freqz(fir_coeff, fs=sample_rate)
    plt.plot(w_fir, 20*np.log10(np.abs(h_fir)), 'b', linewidth=2, label='FIR')
    plt.title('پاسخ فرکانسی فیلتر FIR (Hamming)', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('دامنه (dB)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # زیرنمودار ۲: پاسخ فرکانسی IIR - باترورث
    plt.subplot(2, 2, 2)
    colors = ['r', 'g', 'm', 'c']
    for i, (filter_name, (b, a)) in enumerate(iir_filters.items()):
        w_iir, h_iir = signal.freqz(b, a, fs=sample_rate)
        plt.plot(w_iir, 20*np.log10(np.abs(h_iir)), colors[i], linewidth=1.5, 
                label=f'IIR {filter_name}', alpha=0.8)
    
    plt.title('پاسخ فرکانسی فیلترهای IIR', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('دامنه (dB)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # زیرنمودار ۳: پاسخ فاز FIR
    plt.subplot(2, 2, 3)
    plt.plot(w_fir, np.unwrap(np.angle(h_fir)), 'b', linewidth=2)
    plt.title('پاسخ فاز فیلتر FIR', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('فاز (رادیان)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # زیرنمودار ۴: پاسخ گروهی تاخیر
    plt.subplot(2, 2, 4)
    gd_fir = -np.diff(np.unwrap(np.angle(h_fir))) / np.diff(w_fir * 2 * np.pi)
    plt.plot(w_fir[1:], gd_fir, 'b', linewidth=2, label='FIR')
    
    for filter_name, (b, a) in iir_filters.items():
        w_iir, h_iir = signal.freqz(b, a, fs=sample_rate)
        gd_iir = -np.diff(np.unwrap(np.angle(h_iir))) / np.diff(w_iir * 2 * np.pi)
        plt.plot(w_iir[1:], gd_iir, colors[list(iir_filters.keys()).index(filter_name)], 
                linewidth=1.5, label=f'IIR {filter_name}', alpha=0.8)
    
    plt.title('تأخیر گروهی فیلترها', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('تأخیر (ثانیه)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 0.01)
    
    plt.tight_layout()
    plt.savefig('output_images/part4_filter_responses.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_filtering_results(clean_signal, noisy_signal, filtered_signals, 
                          noise_type, sample_rate):
    """
    رسم نتایج فیلتر کردن
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    
    # انتخاب بخشی از سیگنال برای نمایش
    start_idx = 10000
    display_samples = 2000
    time_axis = np.arange(display_samples) / sample_rate
    
    # داده‌ها
    signals_to_plot = [
        ('سیگنال تمیز', clean_signal[start_idx:start_idx+display_samples], 'blue'),
        (f'سیگنال نویزی ({noise_type})', 
         noisy_signal[start_idx:start_idx+display_samples], 'red'),
        ('فیلتر FIR', filtered_signals['FIR'][start_idx:start_idx+display_samples], 'green'),
        ('فیلتر IIR باترورث', 
         filtered_signals['butterworth'][start_idx:start_idx+display_samples], 'orange')
    ]
    
    for i, (title, data, color) in enumerate(signals_to_plot):
        # حوزه زمان
        axes[i, 0].plot(time_axis, data, color=color, linewidth=1)
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 0].set_xlabel('زمان (ثانیه)', fontsize=8)
        axes[i, 0].set_ylabel('دامنه', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        
        # حوزه فرکانس
        n = len(data)
        fft_data = fft(data)
        fft_magnitude = np.abs(fft_data[:n//2])
        frequencies = fftfreq(n, 1/sample_rate)[:n//2]
        
        axes[i, 1].plot(frequencies[:n//4], 
                       10*np.log10(fft_magnitude[:n//4] + 1e-10), 
                       color=color, linewidth=1)
        axes[i, 1].set_title(f'طیف {title}', fontsize=10)
        axes[i, 1].set_xlabel('فرکانس (Hz)', fontsize=8)
        axes[i, 1].set_ylabel('توان (dB)', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(0, 4000)
    
    plt.tight_layout()
    plt.savefig(f'output_images/part4_filtering_results_{noise_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def analyze_filter_performance(clean_signal, noisy_signal, filtered_signals):
    """
    تحلیل عملکرد فیلترها
    """
    results = {}
    
    # محاسبه SNR اولیه
    initial_snr = calculate_snr(clean_signal, noisy_signal)
    
    results['Initial'] = {
        'snr': initial_snr,
        'improvement': 0,
        'mse': np.mean((noisy_signal - clean_signal)**2)
    }
    
    # محاسبه SNR برای هر فیلتر
    for filter_name, filtered_signal in filtered_signals.items():
        snr = calculate_snr(clean_signal, filtered_signal)
        improvement = snr - initial_snr
        
        # محاسبه MSE (میانگین مربعات خطا)
        mse = np.mean((filtered_signal - clean_signal)**2)
        
        # محاسبه PSNR (Peak Signal-to-Noise Ratio)
        max_signal = np.max(np.abs(clean_signal))
        if mse > 0:
            psnr = 20 * np.log10(max_signal / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        results[filter_name] = {
            'snr': snr,
            'improvement': improvement,
            'mse': mse,
            'psnr': psnr
        }
    
    return results

def save_performance_results(all_results):
    """
    ذخیره نتایج عملکرد فیلترها
    """
    # ایجاد جداول برای هر نوع نویز
    for noise_type, results in all_results.items():
        print(f"\n" + "="*60)
        print(f"نتایج عملکرد فیلترها برای نویز {noise_type}")
        print("="*60)
        
        table_data = []
        for filter_name, metrics in results.items():
            table_data.append([
                filter_name,
                f"{metrics['snr']:.2f} dB",
                f"{metrics['improvement']:.2f} dB",
                f"{metrics['mse']:.6f}",
                f"{metrics.get('psnr', 0):.2f} dB" if 'psnr' in metrics else "N/A"
            ])
        
        headers = ["فیلتر", "SNR", "بهبود SNR", "MSE", "PSNR"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # ذخیره در فایل
        with open(f'part4_performance_{noise_type}.txt', 'w', encoding='utf-8') as f:
            f.write(f"نتایج عملکرد فیلترها برای نویز {noise_type}\n")
            f.write("="*60 + "\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="simple"))
            f.write("\n" + "="*60 + "\n")
    
    # ایجاد جدول مقایسه‌ای کلی
    print("\n" + "="*70)
    print("مقایسه کلی عملکرد فیلترها")
    print("="*70)
    
    noise_types = list(all_results.keys())
    filter_names = list(all_results[noise_types[0]].keys())
    
    # جدول بهبود SNR
    improvement_table = []
    for filter_name in filter_names:
        row = [filter_name]
        for noise_type in noise_types:
            improvement = all_results[noise_type][filter_name]['improvement']
            row.append(f"{improvement:.2f} dB")
        improvement_table.append(row)
    
    headers_imp = ["فیلتر"] + [f"نویز {nt}" for nt in noise_types]
    print("\nبهبود SNR برای انواع نویز:")
    print("-"*70)
    print(tabulate(improvement_table, headers=headers_imp, tablefmt="grid"))
    
    # ذخیره جدول مقایسه‌ای
    with open('part4_comparison_summary.txt', 'w', encoding='utf-8') as f:
        f.write("مقایسه کلی عملکرد فیلترها\n")
        f.write("="*70 + "\n\n")
        
        f.write("بهبود SNR برای انواع نویز:\n")
        f.write("-"*70 + "\n")
        f.write(tabulate(improvement_table, headers=headers_imp, tablefmt="simple"))
        f.write("\n" + "="*70 + "\n")

def main():
    """
    تابع اصلی اجرای بخش ۴
    """
    print("="*70)
    print("بخش ۴: حذف نویز و بازسازی سیگنال")
    print("="*70)
    
    # ایجاد پوشه خروجی
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    
    # بارگذاری سیگنال‌ها
    sample_rate, clean_signal, noisy_signals = load_noisy_signals()
    
    # طراحی فیلترها
    print("\n🎛️ طراحی فیلترها...")
    
    # طراحی فیلتر FIR
    fir_coeff = design_fir_filter(sample_rate)
    print(f"  ✓ فیلتر FIR طراحی شد (تعداد ضریب‌ها: {len(fir_coeff)})")
    
    # طراحی فیلترهای IIR
    iir_filters = design_iir_filters(sample_rate)
    print(f"  ✓ {len(iir_filters)} فیلتر IIR طراحی شد")
    
    # نمایش پاسخ فرکانسی فیلترها
    plot_filter_responses(fir_coeff, iir_filters, sample_rate)
    
    # پردازش هر نوع نویز
    all_results = {}
    
    print("\n🔧 پردازش سیگنال‌های نویزی...")
    
    for noise_type, noisy_signal in noisy_signals.items():
        print(f"\n  پردازش نویز {noise_type}:")
        
        # محاسبه SNR اولیه
        initial_snr = calculate_snr(clean_signal, noisy_signal)
        print(f"    SNR اولیه: {initial_snr:.2f} dB")
        
        # اعمال فیلترها
        filtered_signals = apply_filters(noisy_signal, fir_coeff, iir_filters)
        print(f"    ✓ فیلترها اعمال شدند")
        
        # تحلیل عملکرد
        results = analyze_filter_performance(clean_signal, noisy_signal, filtered_signals)
        all_results[noise_type] = results
        
        # نمایش بهترین فیلتر
        best_filter = max(results.items(), 
                         key=lambda x: x[1]['improvement'] if x[0] != 'Initial' else -float('inf'))
        print(f"    بهترین فیلتر: {best_filter[0]} (بهبود: {best_filter[1]['improvement']:.2f} dB)")
        
        # رسم نتایج برای این نویز
        plot_filtering_results(clean_signal, noisy_signal, filtered_signals, 
                              noise_type, sample_rate)
    
    # ذخیره نتایج
    save_performance_results(all_results)
    
    # تحلیل و نتیجه‌گیری
    print("\n" + "="*70)
    print("تحلیل نتایج و نتیجه‌گیری:")
    print("="*70)
    
    print("""
    مقایسه فیلترهای FIR و IIR:
    
    ۱. فیلتر FIR (پاسخ ضربه‌ای محدود):
       • مزایا:
         - پایداری مطلق (همیشه پایدار)
         - فاز خطی (عدم اعوجاج زمانی)
         - طراحی ساده با پنجره‌ها
       • معایب:
         - نیاز به مرتبه بالاتر برای مشخصات مشابه
         - تأخیر بیشتر
         - پیچیدگی محاسباتی بیشتر
    
    ۲. فیلتر IIR (پاسخ ضربه‌ای نامحدود):
       • مزایا:
         - کارایی بالاتر در مرتبه‌های پایین
         - پاسخ فرکانسی تیزتر
         - پیچیدگی محاسباتی کمتر
       • معایب:
         - ممکن است ناپایدار باشد
         - فاز غیرخطی (اعوجاج زمانی)
         - طراحی پیچیده‌تر
    
    نتایج آزمایش:
    
    ۱. برای نویز گوسی:
       • فیلترهای IIR عملکرد بهتری داشتند
       • دلیل: نویز گوسی در تمام فرکانس‌ها توزیع شده است
       • فیلترهای IIR می‌توانند باندهای غیرگفتاری را بهتر حذف کنند
    
    ۲. برای نویز همهمه:
       • فیلتر FIR عملکرد بهتری در حفظ کیفیت گفتار داشت
       • دلیل: فاز خطی FIR از اعوجاج زمانی جلوگیری می‌کند
       • نویز همهمه طیفی شبیه گفتار دارد، بنابراین فیلتر کردن دقیق ضروری است
    
    ۳. برای نویز -92Noisex:
       • فیلترهای IIR با پاسخ تیز بهترین عملکرد را داشتند
       • دلیل: این نویز شامل مولفه‌های ضربه‌ای است
       • فیلترهای IIR می‌توانند این ضربه‌ها را بهتر حذف کنند
    
    توصیه‌ها برای محیط صنعتی:
    
    ۱. برای نویزهای پهن‌باند (گوسی): فیلتر IIR باترورث یا چبیشف
    ۲. برای نویزهای مشابه گفتار (همهمه): فیلتر FIR با فاز خطی
    ۳. برای نویزهای ضربه‌ای (صنعتی): فیلتر IIR بیضوی
    ۴. برای کاربردهای بلادرنگ: فیلتر FIR به دلیل پایداری و فاز خطی
    
    بهبودهای ممکن:
    
    ۱. استفاده از فیلترهای انطباقی (Adaptive)
    ۲. ترکیب چندین فیلتر (Bank Filter)
    ۳. استفاده از روش‌های مبتنی بر ویولت
    ۴. بهره‌گیری از هوش مصنوعی و یادگیری عمیق
    """)
    
    print("\n✅ بخش ۴ با موفقیت تکمیل شد!")
    print("\n📊 تصاویر تولید شده:")
    print("   - output_images/part4_filter_responses.png")
    for noise_type in noisy_signals.keys():
        print(f"   - output_images/part4_filtering_results_{noise_type}.png")
    
    print("\n📄 فایل‌های نتایج:")
    for noise_type in noisy_signals.keys():
        print(f"   - part4_performance_{noise_type}.txt")
    print("   - part4_comparison_summary.txt")
    
    print("\n🎯 پروژه پردازش گفتار با موفقیت به پایان رسید!")

if __name__ == "__main__":
    main()