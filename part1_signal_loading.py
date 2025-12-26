"""
بخش ۱: خواندن و نمایش سیگنال گفتار
تمرینات درس پردازش گفتار - سری اول
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os
from tabulate import tabulate

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def load_and_analyze_signal(filename='sound1.wav'):
    """
    خواندن و تحلیل سیگنال صوتی
    """
    try:
        # خواندن فایل صوتی
        sample_rate, audio_data = wav.read(filename)
        
        # اگر فایل استریو باشد، به مونو تبدیل می‌کنیم
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # نرمال‌سازی سیگنال
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        print(f"✅ فایل صوتی '{filename}' با موفقیت خوانده شد.")
        
    except FileNotFoundError:
        print(f"⚠️ فایل '{filename}' یافت نشد. یک سیگنال نمونه تولید می‌کنیم...")
        
        # تولید یک سیگنال نمونه برای تست
        sample_rate = 16000  # نرخ نمونه‌برداری استاندارد برای گفتار
        duration = 3  # مدت زمان بر حسب ثانیه
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # تولید یک سیگنال گفتار شبیه‌سازی شده
        freq1 = 100  # فرکانس پایه
        freq2 = 200  # فرکانس هارمونیک اول
        freq3 = 300  # فرکانس هارمونیک دوم
        freq4 = 800  # فرکانس فرمانت
        
        audio_data = (0.5 * np.sin(2 * np.pi * freq1 * t) +
                      0.3 * np.sin(2 * np.pi * freq2 * t) +
                      0.2 * np.sin(2 * np.pi * freq3 * t) +
                      0.4 * np.sin(2 * np.pi * freq4 * t) *
                      np.exp(-0.5 * (t - duration/2)**2))
        
        # نرمال‌سازی
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    return sample_rate, audio_data

def calculate_signal_properties(sample_rate, audio_data):
    """
    محاسبه مشخصات سیگنال
    """
    # محاسبه طول سیگنال
    signal_length = len(audio_data) / sample_rate  # طول بر حسب ثانیه
    num_samples = len(audio_data)
    
    # محاسبه مقادیر آماری
    mean_value = np.mean(audio_data)
    std_value = np.std(audio_data)
    max_value = np.max(audio_data)
    min_value = np.min(audio_data)
    
    return {
        'sample_rate': sample_rate,
        'num_samples': num_samples,
        'signal_length': signal_length,
        'mean': mean_value,
        'std': std_value,
        'max': max_value,
        'min': min_value
    }

def plot_signal_time_domain(time_axis, audio_data, sample_rate):
    """
    رسم سیگنال در حوزه زمان
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # نمودار کامل سیگنال
    ax1.plot(time_axis, audio_data, color='blue', linewidth=0.5)
    ax1.set_title('سیگنال گفتار در حوزه زمان (کل سیگنال)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('زمان (ثانیه)', fontsize=10)
    ax1.set_ylabel('دامنه', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_axis[-1])
    
    # نمودار بخشی از سیگنال (برای جزئیات بیشتر)
    ax2.plot(time_axis[:min(2000, len(audio_data))], 
             audio_data[:min(2000, len(audio_data))], 
             color='red', linewidth=1)
    ax2.set_title(f'سیگنال گفتار در حوزه زمان ({min(2000, len(audio_data))} نمونه اول)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('زمان (ثانیه)', fontsize=10)
    ax2.set_ylabel('دامنه', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output_images/part1_signal_time_domain.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # نمایش اسیلوگرام (نمایش شدت بر حسب زمان)
    plt.figure(figsize=(12, 4))
    plt.specgram(audio_data, Fs=sample_rate, NFFT=512, noverlap=256)
    plt.title('اسیلوگرام سیگنال گفتار', fontsize=12, fontweight='bold')
    plt.xlabel('زمان (ثانیه)', fontsize=10)
    plt.ylabel('فرکانس (Hz)', fontsize=10)
    plt.colorbar(label='توان (dB)')
    plt.savefig('output_images/part1_signal_spectrogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_table(properties):
    """
    ذخیره نتایج در جدول
    """
    table_data = [
        ["نرخ نمونه‌برداری", f"{properties['sample_rate']} Hz"],
        ["تعداد نمونه‌ها", f"{properties['num_samples']:,}"],
        ["طول سیگنال", f"{properties['signal_length']:.3f} ثانیه"],
        ["میانگین دامنه", f"{properties['mean']:.6f}"],
        ["انحراف معیار دامنه", f"{properties['std']:.6f}"],
        ["حداکثر دامنه", f"{properties['max']:.6f}"],
        ["حداقل دامنه", f"{properties['min']:.6f}"]
    ]
    
    print("\n" + "="*60)
    print("نتایج تحلیل سیگنال")
    print("="*60)
    print(tabulate(table_data, headers=["پارامتر", "مقدار"], tablefmt="grid"))
    print("="*60)
    
    # ذخیره جدول در فایل
    with open('part1_signal_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("نتایج تحلیل سیگنال گفتار - بخش ۱\n")
        f.write("="*60 + "\n")
        f.write(tabulate(table_data, headers=["پارامتر", "مقدار"], tablefmt="simple"))
        f.write("\n" + "="*60 + "\n")
    
    print("\n✅ نتایج در فایل 'part1_signal_analysis_results.txt' ذخیره شد.")

def main():
    """
    تابع اصلی اجرای بخش ۱
    """
    print("="*60)
    print("بخش ۱: خواندن و نمایش سیگنال گفتار")
    print("="*60)
    
    # ایجاد پوشه خروجی
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    
    # 1a. خواندن فایل صوتی
    sample_rate, audio_data = load_and_analyze_signal()
    
    # 1b. محاسبه مشخصات سیگنال
    properties = calculate_signal_properties(sample_rate, audio_data)
    
    # نمایش نتایج
    save_results_table(properties)
    
    # 1a. نمایش سیگنال در حوزه زمان
    time_axis = np.arange(len(audio_data)) / sample_rate
    plot_signal_time_domain(time_axis, audio_data, sample_rate)
    
    # 1c. توضیح اهمیت نرخ نمونه‌برداری
    print("\n" + "="*60)
    print("توضیح اهمیت نرخ نمونه‌برداری در پردازش گفتار:")
    print("="*60)
    print("""
    ۱. طبق قضیه نایکویست-شانون: برای بازسازی صحیح یک سیگنال پیوسته از نمونه‌های 
       گسسته، نرخ نمونه‌برداری باید حداقل دو برابر بالاترین فرکانس موجود در سیگنال باشد.
       
    ۲. محدوده فرکانسی گفتار انسان: 
       - گفتار معمولی: ۳۰۰-۳۴۰۰ هرتز
       - گفتار با کیفیت بالا: تا ۸۰۰۰ هرتز
       
    ۳. نرخ نمونه‌برداری استاندارد:
       - تلفن: ۸۰۰۰ هرتز (پوشش تا ۴۰۰۰ هرتز)
       - صوت دیجیتال: ۴۴۱۰۰ هرتز
       - گفتار با کیفیت بالا: ۱۶۰۰۰ هرتز
       
    ۴. جلوگیری از پدیده علیاس (Aliasing): 
       اگر نرخ نمونه‌برداری کافی نباشد، فرکانس‌های بالا به صورت فرکانس‌های پایین‌تر
       ظاهر می‌شوند و باعث اعوجاج می‌شوند.
       
    ۵. تأثیر بر حجم داده‌ها: 
       نرخ نمونه‌برداری بالاتر = حجم داده‌های بیشتر = نیاز به پهنای باند بیشتر
       
    ۶. تأثیر بر کیفیت پردازش: 
       نرخ نمونه‌برداری مناسب امکان تحلیل دقیق‌تر ویژگی‌های فرکانسی را فراهم می‌کند.
       
    ۷. در این پروژه: 
       نرخ نمونه‌برداری {} هرتز امکان تحلیل فرکانس‌های تا {} هرتز را فراهم می‌کند.
    """.format(properties['sample_rate'], properties['sample_rate']/2))
    
    print("\n✅ بخش ۱ با موفقیت تکمیل شد!")
    print("📊 تصاویر تولید شده:")
    print("   - output_images/part1_signal_time_domain.png")
    print("   - output_images/part1_signal_spectrogram.png")
    print("📄 فایل نتایج:")
    print("   - part1_signal_analysis_results.txt")

if __name__ == "__main__":
    main()