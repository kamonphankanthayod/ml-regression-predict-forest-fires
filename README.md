# ml-regression-predict-forest-fires
This is part of the course cs372.

# Predict Forest Fires Area (Linear Regression)

## เกี่ยวกับโปรเจกต์ (About This Project)
โปรเจกต์นี้เป็นการนำโมเดล Machine Learning อัลกอริทึม **Linear Regression** มาใช้สำหรับทำนายขนาดพื้นที่ป่าที่ถูกไฟไหม้ (`area`) โดยอิงจากปัจจัยทางอุตุนิยมวิทยาและตัวชี้วัดความชื้นของป่า 

โปรเจกต์นี้อ้างอิงและถูกพัฒนาเพิ่มเติม (Amplify) จากไฟล์ตัวอย่าง `[linear_regression_assignment_for_students.ipynb](https://colab.research.google.com/drive/1TiEk1nXY-6CD9nOQfmkentShBm9FIFUn?usp=sharing)` โดยจัดทำขึ้นเพื่อการศึกษาในรายวิชา **CS372** เท่านั้น

**ผู้จัดทำ:** กมลพันธ์ กันธายอด 6609520116

---

## โครงสร้างไฟล์ (Repository Structure)
ใน Repository นี้ประกอบไปด้วยไฟล์ดังต่อไปนี้:

* **`forestfires.csv`**: ชุดข้อมูลหลัก (Dataset) ที่ประกอบไปด้วยข้อมูล 516 แถว ใช้สำหรับกระบวนการ Train และ Evaluate โมเดล
* **`forestfires2.csv`**: ชุดข้อมูลภายนอก (External Dataset) ที่แยกออกมาต่างหากเพื่อจำลองสถานการณ์การทำนายข้อมูลที่โมเดลไม่เคยเห็นมาก่อน (Unseen Data)
* **`linear_regression_forestfires.ipynb`**: Jupyter Notebook ไฟล์หลักที่บรรจุ Source Code ของโปรเจกต์ ตั้งแต่การโหลดข้อมูลไปจนถึงการทำนายผล

---

## ข้อมูลที่ใช้ (Dataset Features)
ข้อมูลชุดนี้ประกอบไปด้วยปัจจัยต่างๆ ดังนี้:
* **Spatial/Temporal:** `X`, `Y` (พิกัด), `month`, `day` (เดือนและวัน)
* **FWI System Components:** `FFMC`, `DMC`, `DC`, `ISI` (ดัชนีความชื้นและสภาพอากาศที่มีผลต่อไฟป่า)
* **Meteorological Data:** `temp` (อุณหภูมิ), `RH` (ความชื้นสัมพัทธ์), `wind` (ความเร็วลม), `rain` (ปริมาณน้ำฝน)
* **Target Variable:** `area` (พื้นที่ป่าที่ถูกไฟไหม้)

> **หมายเหตุ:** เนื่องจากตัวแปรเป้าหมาย (`area`) มีการกระจายตัวแบบเบ้ขวาอย่างรุนแรง (Highly Skewed) โปรเจกต์นี้จึงมีการใช้การแปลงค่าแบบ **Log Transform (`np.log1p`)** เพื่อให้ข้อมูลมีการกระจายตัวเข้าใกล้โค้งปกติ (Normal Distribution) มากขึ้น ซึ่งเป็นผลดีต่อโมเดล Linear Regression

---

## ขั้นตอนการทำงาน (Methodology Pipeline)
กระบวนการใน Notebook ถูกแบ่งออกเป็นขั้นตอนดังนี้:

1. **Data Loading & EDA:** โหลดข้อมูลและแสดง Histogram เพื่อดูการกระจายตัวของข้อมูลพื้นที่ไฟไหม้
2. **Data Preprocessing:**
   * ใช้ **Log Transform** จัดการกับ Outliers ของคอลัมน์ `area`
   * ใช้ **One-Hot Encoding** แปลงข้อมูลเชิงกลุ่ม (`month`, `day`) ให้เป็นข้อมูลตัวเลข 
3. **Train/Test Split:** แบ่งข้อมูลออกเป็นชุด Train (90%) และ Test (10%)
4. **Cross Validation & Model Tuning:** * สร้าง `Pipeline` รวมขั้นตอนการทำ `StandardScaler` และ `LinearRegression` เข้าด้วยกัน
   * ใช้ `GridSearchCV` (10-Fold CV) เพื่อหาพารามิเตอร์ที่ดีที่สุด (`fit_intercept`)
5. **Model Evaluation:** ประเมินความแม่นยำของโมเดลด้วยค่า **RMSE** และ **R² (R-Squared)** บน Test Set
6. **Equation Extraction:** ดึงค่าสัมประสิทธิ์ (Coefficients) และจุดตัดแกน (Intercept) ออกมาสร้างเป็นสมการคณิตศาสตร์ ทั้งในรูปแบบ **Scaled** และการแปลงกลับเป็น **Original Scale** เพื่อวิเคราะห์ทิศทางและน้ำหนักของแต่ละฟีเจอร์
7. **External Data Prediction:** นำโมเดลที่ฝึกสอนเสร็จแล้ว ไปทดลองทำนายผลลัพธ์จากข้อมูล `forestfires2.csv` และแปลงค่ากลับจาก Log เป็นพื้นที่แบบปกติ

---

## วิธีการรันโปรเจกต์ (How to Run)
1. Clone repository นี้ลงในเครื่องของคุณ
2. แนะนำให้สร้าง Virtual Environment และติดตั้งไลบรารีที่จำเป็น:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
