import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px


st.markdown(
    """
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #333333;
            color: white;
        }
        .Widget>label {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Fungsi untuk perhitungan AHP
def ahp_attributes(ahp_df):
    sum_array = np.array(ahp_df.sum(numeric_only=True))
    cell_by_sum = ahp_df.drop(["Kriteria"], axis=1).div(sum_array, axis=1)
    priority_df = pd.DataFrame(cell_by_sum.mean(axis=1),
                               index=ahp_df.index, columns=['priority index'])
    return priority_df



# Fungsi untuk CR
def consistency_ratio(priority_index, ahp_df):
    # Check for consistency
    consistency_df = ahp_df.drop(ahp_df.columns[[0]], axis=1).multiply(np.array(priority_index.loc['priority index']), axis=1)
    consistency_df['sum_of_col'] = consistency_df.sum(axis=1)
    
    # To find lambda max
    lambda_max_df = consistency_df['sum_of_col'].div(np.array(priority_index.transpose()['priority index']), axis=0)
    lambda_max = lambda_max_df.mean()
    
    # To find the consistency index
    consistency_index = round((lambda_max - len(ahp_df.index)) / (len(ahp_df.index) - 1), 3)
    print(f'The Consistency Index is: {consistency_index}')
    
    # To find the consistency ratio
    consistency_ratio = round(consistency_index / random_matrix[len(ahp_df.index)], 3)
    print(f'The Consistency Ratio is: {consistency_ratio}')
    
    if consistency_ratio < 0.1:
        print('The model is consistent')
    else:
        print('The model is not consistent')
    return consistency_index, consistency_ratio

#Fungsi untuk Input AHP Dhafina's Version
def create_comparison_matrix(criteria):
    n = len(criteria)
    comparison_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                comparison_matrix[i][j] = 1  # Set diagonal to 1
            elif comparison_matrix[i][j] == 0:  # Check if element is already filled
                key = f"select_{i}_{j}"  # Generate a unique key
                st.subheader(f"Perbandingan antara '{criteria[i]}' dan '{criteria[j]}'")
                decision = st.selectbox(f"Pilih kriteria yang lebih penting:", (f"{criteria[i]}", f"{criteria[j]}"), key=key)
                if decision == criteria[i]:
                    value = st.slider(f"Masukkan nilai perbandingan antara '{criteria[i]}' dan '{criteria[j]}' (1-9): ", 1, 9)
                    if value < 1 or value > 9:
                        st.warning("Nilai harus berada dalam rentang 1-9.")
                        j -= 1
                        continue
                    comparison_matrix[i][j] = value
                    comparison_matrix[j][i] = 1 / value
                else:
                    value = st.slider(f"Masukkan nilai perbandingan antara '{criteria[j]}' dan '{criteria[i]}' (1-9): ", 1, 9)
                    if value < 1 or value > 9:
                        st.warning("Nilai harus berada dalam rentang 1-9.")
                        j -= 1
                        continue
                    comparison_matrix[j][i] = value
                    comparison_matrix[i][j] = 1 / value

    st.success("Input perbandingan kriteria telah selesai.")
    return comparison_matrix

criteria = ["Harga", "Kamera", "RAM", "ROM", "Baterai", "Processor"]

alternatives = ["Openess", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


# Fungsi untuk menghitung nilai CI
def calculate_ci(eigenvalue, n):
    return (eigenvalue - n) / (n - 1)

# Fungsi untuk menghitung nilai CR
def calculate_cr(ci, ri):
    return ci / ri

# Fungsi untuk menghitung eigenvalue dan eigenvector menggunakan metode eigen numpy
def calculate_eigen(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue = max(eigenvalues)
    return max_eigenvalue, eigenvectors[:, np.argmax(eigenvalues)]

# Fungsi untuk menghitung konsistensi ratio
def calculate_consistency_ratio(matrix_df):
    n = len(matrix_df)
    matrix = matrix_df.to_numpy()
    max_eigenvalue, eigenvector = calculate_eigen(matrix)
    ci = calculate_ci(max_eigenvalue, n)

    # Referensi indeks konsistensi acak (RI)
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_dict[n]

    cr = calculate_cr(ci, ri)

    if cr <= 0.1:
        st.write("Matriks perbandingan konsisten.")
    else:
        st.write("Matriks perbandingan tidak konsisten.")

    return cr



# Fungsi untuk menampilkan detail
def show_detail():
    st.title('Detail')
    st.write("""
        Bobot distribusi yang digunakan dalam prediksi kepribadian:

        | Kriteria          | Openness | Conscientiousness | Extraversion | Agreeableness | Neuroticism |
        |-------------------|----------|-------------------|--------------|---------------|-------------|
        | Budget            | 0.2077   | 0.1676            | 0.2073       | 0.1990        | 0.2182      |
        | Camera            | 0.1846   | 0.2427            | 0.1635       | 0.1713        | 0.2379      |
        | RAM               | 0.2198   | 0.1821            | 0.1771       | 0.2186        | 0.2024      |
        | ROM               | 0.1973   | 0.2288            | 0.1639       | 0.1716        | 0.2384      |
        | Battery           | 0.2099   | 0.1882            | 0.1845       | 0.2214        | 0.1960      |
        | Processor         | 0.2003   | 0.1712            | 0.2039       | 0.2079        | 0.2166      |
    """)


    # Data bobot distribusi
    data = {
        "Kriteria": ["Budget", "Camera", "RAM", "ROM", "Battery", "Processor"],
        "Openness": [0.2077, 0.1846, 0.2198, 0.1973, 0.2099, 0.2003],
        "Conscientiousness": [0.1676, 0.2427, 0.1821, 0.2288, 0.1882, 0.1712],
        "Extraversion": [0.2073, 0.1635, 0.1771, 0.1639, 0.1845, 0.2039],
        "Agreeableness": [0.1990, 0.1713, 0.2186, 0.1716, 0.2214, 0.2079],
        "Neuroticism": [0.2182, 0.2379, 0.2024, 0.2384, 0.1960, 0.2166]
    }

    # Membuat DataFrame
    df = pd.DataFrame(data)

    # Menggabungkan data bobot distribusi untuk plot
    df_melted = df.melt(id_vars=["Kriteria"], var_name="Personality", value_name="Bobot")

    # Membuat bar chart interaktif
    fig = px.bar(df_melted, x="Kriteria", y="Bobot", color="Personality", barmode="group", color_discrete_sequence=['#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557'])

    # Menampilkan plot
    st.plotly_chart(fig)
    
def consistency_ratio(priority_index, ahp_df):
    priority_index = priority_index.transpose()
    random_matrix = {
        1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32,
        8: 1.14, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56,
        14: 1.57, 15: 1.59, 16: 1.605, 17: 1.61, 18: 1.615, 19: 1.62, 20: 1.625
    }
    # Check for consistency
    consistency_df = ahp_df.drop(df1.columns[[0]], axis=1).multiply(np.array(priority_index.loc['priority index']), axis=1)
    consistency_df['sum_of_col'] = consistency_df.sum(axis=1)
    # To find lambda max
    lambda_max_df = consistency_df['sum_of_col'].div(np.array(priority_index.transpose()['priority index']), axis=0)
    lambda_max = lambda_max_df.mean()
    # To find the consistency index
    consistency_index = round((lambda_max - len(ahp_df.index)) / (len(ahp_df.index) - 1), 3)
    print(f'The Consistency Index is: {consistency_index}')
    # To find the consistency ratio
    consistency_ratio = round(consistency_index / random_matrix[len(ahp_df.index)], 3)
    print(f'The Consistency Ratio is: {consistency_ratio}')
    if consistency_ratio < 0.1:
        print('The model is consistent')
    else:
        print('The model is not consistent')

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.sidebar.title('Menu')
    page = st.sidebar.selectbox("Choose a page", ["Beranda", "Prediction AHP", "Definition", "Detail"])
    if page == "Beranda":
        st.markdown(
        """
        <style>
            @keyframes moveInFromTop {
                from {
                    transform: translateY(-100%);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }

            @keyframes moveInFromBottom {
                from {
                    transform: translateY(100%);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }

            h1 {
                animation: moveInFromTop 1s ease-out;
            }

            .subtitle {
                animation: moveInFromBottom 1s ease-out;
                
            }
        </style>
        """
        , unsafe_allow_html=True
        )

        st.markdown("<br><br><br><br><br><br><br><br><h1 style='text-align: center;'>Welcome to Personality Trait Prediction !</h1><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        
        st.markdown("<p class='subtitle' style='text-align: center; font-size: 20px;'>By Kelompok 14</p>", unsafe_allow_html=True)

    elif page == "Prediction AHP":

        # Comparison Matrix
        st.markdown(
            f"<h1 style='text-align: center;'>AHP Prediction</h1>",
            unsafe_allow_html=True
        )

        comparison_matrix = pd.DataFrame(create_comparison_matrix(criteria), index=criteria, columns=criteria)

        #st.write("\nPairwise Comparison Matrix:")
        #st.write(comparison_matrix)

        # Normalized Pairwise Comparison Matrix
        normalized_matrix = comparison_matrix.div(comparison_matrix.sum(axis=0), axis=1)

        #st.write("\nNormalized Pairwise Comparison Matrix:")
        #st.write(normalized_matrix)

        # Priority Vector
        priority_vector = pd.DataFrame(normalized_matrix.mean(axis=1))
        

        # Ranking of Priorities
        #priority_rank = pd.DataFrame({'Priority Rank': priority_vector.rank(ascending=False).iloc[:,0].tolist()})

        #st.write("\nRanking of Priorities:")
        #st.write(priority_rank)

        # Hitung rasio konsistensi
        st.title('Consistency Ratio')
        consistency_ratio = calculate_consistency_ratio(comparison_matrix)
        st.write("\nConsistency Ratio (CR):", consistency_ratio)


        # Priority Percentage of Personality Traits for Each Criteria
        percent = [
            [0.2073, 0.2427, 0.1771, 0.1639, 0.2099, 0.1712],
            [0.2182, 0.1846, 0.1821, 0.1973, 0.2214, 0.2166],
            [0.2077, 0.1635, 0.2024, 0.2384, 0.1845, 0.2039],
            [0.1676, 0.2379, 0.2186, 0.1716, 0.2214, 0.2003],
            [0.1990, 0.1713, 0.1639, 0.2288, 0.1960, 0.2039]
        ]

        alt_percent = pd.DataFrame(percent, columns=criteria, index=alternatives)
        # Mengubah nama kolom indeks menjadi "Personality"
        alt_percent = alt_percent.rename_axis("Personality")
        priority_vector.columns = ['Priority']

        #st.title('Priority Percentage of Personality Traits for Each Criteria')
        #st.write(alt_percent)

        # Ranking Alternatives
        result = alt_percent.dot(priority_vector)
        st.title('Ranking of Alternatives')
        st.write(result)
         # Tampilkan personality dengan ranking tertinggi
        max_personality = result.idxmax()[0]
        st.markdown(f"""
        <h1 style='text-align: center; font-size: 48px;'>Personality yang terpilih: <span class='animated-text'>{max_personality}</span></h1>
    """, unsafe_allow_html=True)

        # Tambahkan CSS untuk animasi
        st.markdown("""
            <style>
                /* Animasi gradien merah */
                @keyframes colorchange {
                    0% {color: #FF0000;}
                    25% {color: #FF3333;}
                    50% {color: #FF6666;}
                    75% {color: #FF9999;}
                    100% {color: #FFCCCC;}
                }

                /* Menerapkan animasi ke teks */
                .animated-text {
                    animation: colorchange 5s infinite;
                }
            </style>
        """, unsafe_allow_html=True)

    # Page Definition
    elif page == "Definition":
        st.markdown(
            f"<h1 style='text-align: center;'>Personality Trait</h1>",
            unsafe_allow_html=True
        )

        # Daftar kepribadian
        personalities = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        
        # Pilihan drop-down untuk memilih kepribadian
        selected_personality = st.selectbox("Pilih Personality", personalities)

        # Data definisi dan strategi pemasaran
        definition_strategies = {
        "Openness": {
            "Deskripsi": "Openness adalah kecenderungan individu untuk berpikir dan bertindak secara kreatif, terbuka terhadap pengalaman baru, dan memiliki minat yang luas dalam seni, kepemimpinan, atau emosi.",
            "Strategi Pemasaran": "Menggunakan desain yang inovatif dan menarik untuk produk yang dapat meningkatkan kreativitas pelanggan."
        },
        "Conscientiousness": {
            "Deskripsi": "Conscientiousness adalah kecenderungan individu untuk bertindak secara hati-hati, teratur, dan bertanggung jawab dalam segala tindakan.",
            "Strategi Pemasaran": "Menyoroti keandalan dan kualitas produk serta menawarkan jaminan kepuasan pelanggan."
        },
        "Extraversion": {
            "Deskripsi": "Extraversion adalah kecenderungan individu untuk menjadi sosial, percaya diri, dan antusias dalam interaksi sosial.",
            "Strategi Pemasaran": "Mengadakan acara promosi dan aktivitas sosial untuk menarik perhatian pelanggan dan meningkatkan keterlibatan."
        },
        "Agreeableness": {
            "Deskripsi": "Agreeableness adalah kecenderungan individu untuk bersikap ramah, empatik, dan toleran terhadap orang lain.",
            "Strategi Pemasaran": "Menekankan pada layanan pelanggan yang ramah dan menawarkan produk yang mendukung kebutuhan dan nilai-nilai pelanggan."
        },
        "Neuroticism": {
            "Deskripsi": "Neuroticism adalah kecenderungan individu untuk mengalami tingkat kecemasan, ketegangan, dan sensitivitas emosional yang tinggi.",
            "Strategi Pemasaran": "Menyediakan produk atau layanan yang menawarkan rasa aman dan kenyamanan bagi pelanggan."
        }
    }

        # Menampilkan definisi dan strategi pemasaran dalam bentuk tabel
        st.table(definition_strategies[selected_personality])

    elif page == "Detail":
        show_detail()

# Memanggil fungsi utama
if __name__ == '__main__':
    main()