import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'model' else BASE_DIR

OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs') if os.path.exists(os.path.join(BASE_DIR, 'outputs')) else os.path.join(PROJECT_ROOT, 'outputs')
MODEL_PATH = os.path.join(OUTPUTS_DIR, 'models', 'best_cart_model.joblib')
VIZ_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')

TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates') if os.path.exists(os.path.join(BASE_DIR, 'templates')) else os.path.join(PROJECT_ROOT, 'templates')

app = Flask(__name__, template_folder=TEMPLATES_DIR, root_path=BASE_DIR)

# Load model
cart_data = joblib.load(MODEL_PATH)
cart_model = cart_data['model']
feature_names = cart_data['feature_names']
tree = cart_model.tree_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory(VIZ_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form

        age = float(data.get('age', 30))
        sex = int(data.get('sex', 1)) # 0: female, 1: male
        bmi = float(data.get('bmi', 25.0))
        children = int(data.get('children', 0))
        smoker = int(data.get('smoker', 0)) # 0: no, 1: yes
        region = data.get('region', 'northeast').lower()

        # Build feature dict
        input_data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region_northwest': 1 if region == 'northwest' else 0,
            'region_southeast': 1 if region == 'southeast' else 0,
            'region_southwest': 1 if region == 'southwest' else 0
        }

        df_input = pd.DataFrame([input_data])[feature_names]

        # Predict CART
        cart_pred = float(cart_model.predict(df_input)[0])

        # Extract active decision path through tree
        node_indicator = cart_model.decision_path(df_input)
        leaf_id = int(cart_model.apply(df_input)[0])
        active_nodes = [int(i) for i in node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]]

        # Build readable decision trail steps
        steps = []
        name_map = {
            'smoker': 'Status Perokok',
            'bmi': 'Indeks BMI',
            'age': 'Usia Pasien',
            'children': 'Tanggungan Anak'
        }

        for node_id in active_nodes:
            if node_id == leaf_id:
                val = float(tree.value[node_id][0][0])
                steps.append({
                    'node_id': node_id,
                    'is_leaf': True,
                    'title': 'Hasil Simpul Daun (Leaf)',
                    'rule': f'Estimasi biaya final = ${val:,.2f}',
                    'user_val': f'${val:,.2f}',
                    'branch': 'Selesai'
                })
            else:
                feat_idx = tree.feature[node_id]
                feat_name = feature_names[feat_idx]
                thresh = float(tree.threshold[node_id])
                user_val = float(df_input[feat_name].values[0])
                is_left = user_val <= thresh
                feat_label = name_map.get(feat_name, feat_name)

                if feat_name == 'smoker':
                    cond = 'smoker ≤ 0.5 (Bukan Perokok)' if is_left else 'smoker > 0.5 (Perokok Aktif)'
                    val_disp = 'Tidak Merokok' if user_val == 0 else 'Perokok'
                elif feat_name == 'bmi':
                    cond = f'BMI ≤ {thresh:.1f} kg/m²' if is_left else f'BMI > {thresh:.1f} kg/m²'
                    val_disp = f'{user_val:.1f} kg/m²'
                elif feat_name == 'age':
                    cond = f'Usia ≤ {thresh:.0f} th' if is_left else f'Usia > {thresh:.0f} th'
                    val_disp = f'{user_val:.0f} th'
                else:
                    cond = f'{feat_label} ≤ {thresh:.0f}' if is_left else f'{feat_label} > {thresh:.0f}'
                    val_disp = f'{user_val:.0f}'

                steps.append({
                    'node_id': node_id,
                    'is_leaf': False,
                    'feature': feat_name,
                    'title': feat_label,
                    'rule': cond,
                    'user_val': val_disp,
                    'branch': 'Kiri (≤)' if is_left else 'Kanan (>)',
                    'next_node': int(tree.children_left[node_id] if is_left else tree.children_right[node_id])
                })

        # Risk Analysis
        if smoker == 1 and bmi >= 30.0:
            risk_tier = "Sangat Tinggi (High Tier)"
            risk_color = "rose"
            logic_reason = (
                f"Pasien Perokok Aktif dengan BMI {bmi:.1f} kg/m² (≥ 30 Obesitas). "
                f"Pohon CART langsung mengambil cabang kanan pada kedua simpul utama, "
                f"menghasilkan estimasi premi tinggi sebesar ${cart_pred:,.2f}."
            )
        elif smoker == 1:
            risk_tier = "Tinggi (Moderate-High Tier)"
            risk_color = "amber"
            logic_reason = (
                f"Pasien Perokok Aktif dengan BMI {bmi:.1f} kg/m² (< 30 Normal/Gemuk). "
                f"Pohon CART mengambil cabang kanan pada status merokok, lalu cabang kiri pada batas obesitas."
            )
        elif age >= 45:
            risk_tier = "Menengah (Moderate Tier)"
            risk_color = "indigo"
            logic_reason = (
                f"Pasien Bukan Perokok dengan usia {age:.0f} tahun (≥ 43 tahun). "
                f"Pohon CART mengambil cabang kiri pada status merokok, lalu percabangan kanan pada usia matang."
            )
        else:
            risk_tier = "Rendah (Standard Tier)"
            risk_color = "emerald"
            logic_reason = (
                f"Pasien Bukan Perokok dengan usia muda ({age:.0f} th). "
                f"Pohon CART mengarahkan alur ke kelompok premi dasar standar."
            )

        idr_est = cart_pred * 16000

        return jsonify({
            'success': True,
            'cart_prediction': cart_pred,
            'cart_formatted': f"${cart_pred:,.2f}",
            'idr_formatted': f"Rp {idr_est:,.0f}",
            'active_nodes': active_nodes,
            'decision_steps': steps,
            'risk_tier': risk_tier,
            'risk_color': risk_color,
            'logic_reason': logic_reason
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("APLIKASI WEB SIMULASI PREDIKSI ASURANSI (CART REGRESSION)")
    print("Akses web di browser: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
