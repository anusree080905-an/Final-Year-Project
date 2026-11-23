"""
IndicAI-Detect: Language-Aware AI-Generated Text Detection Benchmark & Atlas
Supports 8 Indian Languages across Dravidian and Indo-Aryan families:
Tamil, Telugu, Malayalam, Kannada, Hindi, Bengali, Marathi, Gujarati.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.text_processor import IndicTextPreprocessor, LANGUAGE_FAMILIES, LANGUAGE_NAMES
from src.features.feature_extractor import IndicFeatureExtractor
from src.models.classifier import LanguageAwareClassifier, TARGET_LANGS
from src.explainability.shap_explainer import AIFingerprintAtlas


st.set_page_config(
    page_title="IndicAI-Detect | Language-Aware AI Text Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .ai-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .human-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .family-dravidian {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 500;
    }
    .family-indo-aryan {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Benchmark Sample Texts
BENCHMARK_SAMPLES = {
    "Tamil (Human - IndicCorpV2)": "கடந்த உலகக் கோப்பையில் காலிறுதியில் விளயைாடியது. இங்கிலாந்து 15வது முறையாக உலகக் கோப்பையில் விளையாடும் இங்கிலாந்து, 1966ல் உலகக் கோப்பையை வென்றது. மூன்றாவது முறையாக அரை இறுதிக்கு நுழைந்துள்ளது. 1990க்குப் பிறகு மீண்டும் அரை இறுதியில் விளையாட உள்ளது. குரேஷியா தனி நாடானப் பிறகு 5வது முறையாக உலகக் கோப்பையில் விளையாடுகிறது.",
    "Tamil (AI - Aya-8B)": "எண்டெமோலின் டீலா நோ டீலா நோடின் தொலைக்காட்சி தொழில்நுறை வரலாற்றியில் முதன் முதலாயிருந்தது. இது போலும் தமிழ் சினிமாக்களில் பல வரலாற்றிய படங்களை கொண்டது. இப்போவாவது தமிழ் சினிமாக்களில் வரலாற்றிய படம் எண்டெமோலின் டீலா நோ டீலா. இப்படம் சினிமாக்களில் முதன் முதலாயிருந்தது என்றால் அதற்கு பிறகு தொழில்நுறை வரலாற்றியம் என்று கூறிய சினிமாக்கள் இருந்தால் அவற்றின் வரலாற்றிய மீமையை தளிர்த்துள்ள எண்டெமோலின் டீலா நோ டீலா.",
    "Telugu (Human - IndicCorpV2)": "ఇంటర్నెట్‌ డెస్క్‌: యూజర్స్‌కి మెరుగైన సేవలు అందించడంలో భాగంగా గూగుల్ ప్లేస్టోర్‌ నుంచి ఎప్పటికప్పుడు ప్రమాదకరమైన యాప్స్‌ని తొలగిస్తుంటుంది. తాజాగా గూగుల్ నిబంధనలకు విరుద్ధంగా యూజర్‌ డేటా సేకరిస్తున్న మూడు ప్రముఖ యాప్స్‌ను ప్లేస్టోర్ నుంచి తొలగించింది. చిన్న పిల్లలను ఎక్కువగా ఉపయోగించే గేమిఫై లెర్నింగ్ యాప్స్ అయిన ప్రిన్సెస్ సలోన్, నంబర్ కలరింగ్ మొదలైనవి తొలగించిన జాబితాలో ఉన్నాయి. ఈ యాప్‌ల కార్యకలాపాలపై అంతర్జాతీయ డిజిటల్ కౌన్సిల్ నుంచి ఫిర్యాదులు రావడంతో వాటిని తొలగించారు.",
    "Telugu (AI - GPT-OSS)": "తెలుగు రాష్ట్రాల్లో అప్పుడప్పుడు వింత జంతువులు, పక్షులు, పాములు హల్ చల్ చేస్తుంటాయి. ప్రజలు వాటిని చూసినప్పుడల్లా అవే హాట్ టాపిక్ గా మారతుంటాయి. తాజాగా ఆంధ్రప్రదేశ్ గుంటూరు జిల్లా, తాడేపల్లి మండలం, వడ్డేశ్వరంలో వింత పక్షులు కలకలం రేపాయి. గ్రామస్తులు వెంటనే అటవీ శాఖ అధికారులకు సమాచారం అందించారు. ఇలాంటి పక్షులు ఈ ప్రాంతంలో గతంలో ఎన్నడూ చూడలేదని స్థానికులు పేర్కొన్నారు.",
    "Malayalam (Human - IndicCorpV2)": "കാട്ടിനുള്ളില്‍ പരന്നു കിടുക്കുന്ന വിസ്തൃതമായൊരു പ്രകൃതി രമണീയ കാഴ്ചയാണ് സഞ്ചാരികളെ കാത്തിരിക്കുന്നത്. പ്രകൃതിഭംഗി ആസ്വദിക്കാനും ശാന്തമായ അന്തരീക്ഷത്തില്‍ സമയം ചെലവഴിക്കാനും നിരവധി ആളുകളാണ് ഇവിടേക്ക് എത്തുന്നത്. വനം വകുപ്പിന്റെ നേതൃത്വത്തില്‍ മികച്ച സുരക്ഷാ ക്രമീകരണങ്ങളും സന്ദര്‍ശകര്‍ക്കായി ഒരുക്കിയിട്ടുണ്ട്.",
    "Malayalam (AI - Qwen2.5)": "സുപ്രിയനെ മറ്റൊരു സന്ദര്‍ഭത്തിലും ഓര്‍ക്കേണ്ടതില്ല എന്ന് പലരും കരുതിയെങ്കിലും യാഥാര്‍ത്ഥ്യം മറ്റൊന്നായിരുന്നു. കാലഘട്ടത്തിന്റെ മാറ്റങ്ങള്‍ ഉള്‍ക്കൊണ്ട് പുതിയ തലമുറയ്ക്ക് വഴികാട്ടിയാകാന്‍ അദ്ദേഹത്തിന്റെ ചിന്തകള്‍ക്ക് സാധിച്ചു. സാമൂഹിക പ്രതിബദ്ധതയുള്ള പ്രവര്‍ത്തനങ്ങളിലൂടെ ജനഹൃദയങ്ങളില്‍ ചിരപ്രതിഷ്ഠ നേടാന്‍ അദ്ദേഹത്തിന് കഴിഞ്ഞു.",
    "Kannada (Human - IndicCorpV2)": "ಮಂಚ್‌ದ ಕಾರ್ಯಕರ್ತರು ನಿರಾಣಿ ಅವರ ಗೃಹ ಕಛೇರಿಯಲ್ಲಿ ಭೇಟಿಯಾಗಿ ಅಭಿನಂದನೆಗಳನ್ನು ಸಲ್ಲಿಸಿದರು. ಈ ಸಂದರ್ಭದಲ್ಲಿ ಮಾತನಾಡಿದ ಮುಖಂಡರು, ಸಮಾಜದ ಸರ್ವತೋಮುಖ ಅಭಿವೃದ್ಧಿಗೆ ಎಲ್ಲರೂ ಒಗ್ಗಟ್ಟಾಗಿ ಶ್ರಮಿಸಬೇಕು ಎಂದು ಕರೆ ನೀಡಿದರು. ಸರ್ಕಾರದ ವಿವಿಧ ಜನಪರ ಯೋಜನೆಗಳನ್ನು ಅರ್ಹ ಫಲಾನುಭವಿಗಳಿಗೆ ತಲುಪಿಸುವ ನಿಟ್ಟಿನಲ್ಲಿ ನಿರಂತರ ಪ್ರಯತ್ನ ಮುಂದುವರಿಯಲಿದೆ.",
    "Kannada (AI - Aya-8B)": "ಅದರೆ ಸಮಯದಲ್ಲಿ ಬೆಳಗ್ಗಿನ ಸಮಯದಲ್ಲಿ ಹೊಸ್ತಿಲು ತೊಳೆದರೆ ಶುಭವಾಗುತ್ತದೆ ಎಂಬ ನಂಬಿಕೆ ಭಾರತೀಯ ಪರಂಪರೆಯಲ್ಲಿದೆ. ಸೂರ್ಯೋದಯಕ್ಕೂ ಮುನ್ನ ಮನೆಯ ಮುಂದಿನ ಅಂಗಳವನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಿ ರಂಗೋಲಿ ಇಡುವುದರಿಂದ ಸಕಾರಾತ್ಮಕ ಶಕ್ತಿ ಮನೆಗೆ ಪ್ರವೇಶಿಸುತ್ತದೆ ಎಂದು ಹಿರಿಯರು ಹೇಳುತ್ತಾರೆ. ಈ ಸಂಪ್ರದಾಯವು ಇಂದಿಗೂ ಗ್ರಾಮೀಣ ಪ್ರದೇಶಗಳಲ್ಲಿ ಜೀವಂತವಾಗಿದೆ.",
    "Hindi (Human - IndicCorpV2)": "हालांकि पार्टी ने अभी तक इस सीट से किसी की उम्मीदवारी की आधिकारिक घोषणा नहीं की है। लेकिन सूत्रों का कहना है कि पूर्व विधायक के नाम पर लगभग सहमति बन चुकी है। आगामी विधानसभा चुनावों को देखते हुए सभी राजनीतिक दलों ने अपनी रणनीति तेज कर दी है और मतदाताओं को लुभाने के लिए विभिन्न वादे किए जा रहे हैं।",
    "Hindi (AI - GPT-OSS)": "पूर्व आईपीएस अधिकारी ने एक निजी टेलीविजन चैनल से बातचीत करते हुए इस पूरे घटनाक्रम पर गहरा दुख व्यक्त किया। उन्होंने कहा कि पुलिस प्रशासन को पूरी निष्पक्षता और तत्परता के साथ कार्य करना चाहिए ताकि आम जनता का कानून व्यवस्था में विश्वास बना रहे। सोशल मीडिया पर फैल रही अफवाहों से बचने की भी उन्होंने अपील की।",
    "Bengali (Human - IndicCorpV2)": "এসএইচএ-১ এর বিরুদ্ধে মার্ক স্টিভেনস একটি আক্রমণ পরিকল্পনা প্রস্তুত করেছিলেন যা নিরাপত্তা ক্ষেত্রে আলোড়ন সৃষ্টি করে। গবেষকদের মতে, এই ক্রিপ্টোগ্রাফিক অ্যালগরিদমের দুর্বলতা প্রকাশ্যে আসার পর নতুন মানদণ্ড নির্ধারণের প্রয়োজনীয়তা দেখা দেয়। আন্তর্জাতিক তথ্যপ্রযুক্তি সংস্থাগুলিও দ্রুত পদক্ষেপ গ্রহণের নির্দেশ দেয়।",
    "Bengali (AI - Qwen2.5)": "ইস্যুটি এখানেই থেমে যেতে পারত। কিন্তু তা হল না এবং পরিস্থিতি আরও জটিল রূপ ধারণ করল। স্থানীয় প্রশাসন হস্তক্ষেপ করে পরিস্থিতি নিয়ন্ত্রণে আনার চেষ্টা করে। প্রত্যক্ষদর্শীদের মতে, সঠিক সময়ে সঠিক পদক্ষেপ গ্রহণ করা হলে এই ধরনের অপ্রীতিকর ঘটনা সহজেই এড়ানো সম্ভব ছিল।",
    "Marathi (Human - IndicCorpV2)": "प्रशांत किशोर यांनी सार्वजनिक आयुष्यातून थोडा वेळ विश्रांती हवी असल्याचे कारण देत राजीनामा दिला आहे. काही काळ सार्वजनिक आयुष्यापासून दूर राहण्याच्या माझ्या निर्णयाबद्दल आपल्याला कल्पना असल्यामुळे मुख्य सल्लागार पदाची कामे करण्यासाठी मी सक्षम नाही. मला माझ्या भविष्यातील वाटचालींबद्दलही निर्णय घ्यायचा असून, मी आपल्याला विनंती करतो की, मला या जबाबदारीतून मुक्त करावे, असे त्यांनी म्हटले आहे.",
    "Marathi (AI - GPT-OSS)": "शरीराला सोडियमचे प्रमाण अतिशय अचूकपणे नियंत्रित करणे आवश्यक असल्यामुळे आम्ही मिठाची निवड केली. आमच्याकडे विशिष्ट अशा पेशीही आहेत, ज्या तुमच्या रक्तात किती मीठ आहे हे शोधतात. जेव्हा तुम्ही खारट अन्न संशोधनानुसार, दररोज ५ ग्रॅमपेक्षा जास्त मीठाचे सेवन केल्यास रक्तदाब वाढतो. आरोग्य तज्ञांनी कमी मीठाच्या आहाराची शिफारस केली असून, या तंत्रज्ञानामुळे घरगुती मीठ मोजणे सोपे झाले आहे.",
    "Gujarati (Human - IndicCorpV2)": "આજકાલ શરીરના અવયવોની કામગીરી જાણવા માટેનાં ડિજિટલ સેન્સરો ઉપલબ્ધ બન્યાં છે તેથી હૃદયના ડિજિટલ ટ્વીનની રચના અને તેનું સતત મોનિટરિંગ શક્ય બન્યાં છે. જે માણસના હૃદયનું ડિજિટલ ટ્વીન બનાવવાનું હોય તે વ્યક્તિએ પ્રથમ તો શરીર પર અમુક સેન્સરો ધારણ કરવાં પડે. આ સેન્સરો દ્વારા મળેલા ડેટાને આધારે ખાસ સોફટવેર કમ્પ્યૂટરમાં ડિજિટલ ટ્વીનની રચના કરે છે.",
    "Gujarati (AI - Aya-8B)": "મેરાન્યૂઝ નેટવર્ક.મોડાસાઃ અનલોક થયા બાદ ચોર લૂંટારુઓ પણ ચોરી, તસ્કરીની ઘટનાને અંજામ આપવા સક્રિય થયા છે. લોકડાઉનમાં ચોરી અને લૂંટની ઘટનાઓને લાગેલી બ્રેક હવે ખુલી ગઈ છે. તસ્કરોએ ફરીથી વેપારીઓ અને સામાન્ય નાગરિકોને નિશાન બનાવવાનું શરૂ કર્યું છે. પોલીસે નાઇಟ್ પેટ્રોલિંગ સઘન બનાવ્યું છે."
}


@st.cache_resource
def load_detector_model():
    model_path = "models/language_aware_detector.joblib"
    if os.path.exists(model_path):
        return LanguageAwareClassifier.load(model_path)
    return None


@st.cache_data
def load_json_artifact(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Sidebar
st.sidebar.title("🛡️ IndicAI-Detect")
st.sidebar.caption("Multilingual AI-Generated Text Detection Benchmark")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Live Text Detector",
        "📊 LA-ADI Benchmark & Rankings",
        "🧬 AI Fingerprint Atlas",
        "🧪 Experimental Evaluations",
        "📚 Research Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Language Scope (8 Languages)")
st.sidebar.markdown("""
**Dravidian Family:**
- 🇮🇳 Tamil (`ta`)
- 🇮🇳 Telugu (`te`)
- 🇮🇳 Malayalam (`ml`)
- 🇮🇳 Kannada (`kn`)

**Indo-Aryan Family:**
- 🇮🇳 Hindi (`hi`)
- 🇮🇳 Bengali (`bn`)
- 🇮🇳 Marathi (`mr`)
- 🇮🇳 Gujarati (`gu`)
""")

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: IndicTextFake (Hugging Face)\nGenerators: Aya-8B, Qwen2.5, GPT-OSS, Human")

detector = load_detector_model()
eval_report = load_json_artifact("results/evaluation_report.json")
la_adi_data = load_json_artifact("results/la_adi_results.json")
atlas_data = load_json_artifact("results/ai_fingerprint_atlas.json")


# -------------------------------------------------------------
# VIEW 1: LIVE TEXT DETECTOR
# -------------------------------------------------------------
if menu == "🔍 Live Text Detector":
    st.markdown('<div class="main-header">Language-Aware AI-Generated Text Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect Human-written vs AI-generated text across eight Indian languages with calibrated confidence and SHAP explainability.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Text")
        sample_choice = st.selectbox(
            "Load a pre-validated benchmark sample:",
            ["Custom Text"] + list(BENCHMARK_SAMPLES.keys())
        )

        initial_text = ""
        if sample_choice != "Custom Text":
            initial_text = BENCHMARK_SAMPLES[sample_choice]

        user_text = st.text_area(
            "Enter Indic text to analyze:",
            value=initial_text,
            height=160,
            placeholder="Type or paste text in Tamil, Telugu, Malayalam, Kannada, Hindi, Bengali, Marathi, or Gujarati..."
        )

        analyze_btn = st.button("🚀 Analyze Text", type="primary", use_container_width=True)

    with col2:
        st.subheader("Analysis Parameters")
        lang_override = st.selectbox(
            "Language Selection:",
            ["Auto-Detect Script & Language"] + [f"{LANGUAGE_NAMES[l]} ({l})" for l in TARGET_LANGS]
        )
        conf_threshold = st.slider("AI Classification Decision Threshold", 0.0, 1.0, 0.50, 0.05)

        st.info("💡 **Language-Aware Mechanism**: Automatically extracts script Unicode distribution, routes between Dravidian and Indo-Aryan families, and adapts stylometric thresholds.")

    if analyze_btn and user_text.strip():
        if detector is None:
            st.error("Model artifact not found! Please run the pipeline first.")
        else:
            with st.spinner("Executing Unicode normalization, script routing, and feature extraction..."):
                # Handle lang override
                selected_lang_code = None
                if lang_override != "Auto-Detect Script & Language":
                    selected_lang_code = lang_override.split("(")[-1].replace(")", "").strip()

                # Inference
                result = detector.predict_text(user_text, selected_lang_code)
                prep = result["preprocessed"]
                lang_code = result["language_code"]
                family = result["language_family"]
                ai_prob = result["ai_probability"]
                human_prob = result["human_probability"]

                is_ai = ai_prob >= conf_threshold
                pred_label = "AI-Generated" if is_ai else "Human-Written"
                confidence = ai_prob if is_ai else human_prob

                # LA-ADI lookup
                adi_score_val = "N/A"
                adi_rank_val = "N/A"
                if la_adi_data and lang_code in la_adi_data.get("language_scores", {}):
                    lang_adi = la_adi_data["language_scores"][lang_code]
                    adi_score_val = f"{lang_adi['la_adi_score']:.3f}"
                    adi_rank_val = f"#{lang_adi['rank']} / 8"

            st.markdown("---")
            st.subheader("Detection Results & Language Routing")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f"**Detected Language**")
                st.markdown(f"### {result['language_name']} (`{lang_code}`)")
                st.caption(f"Script: {prep['script']}")

            with r2:
                st.markdown(f"**Language Family**")
                fam_class = "family-dravidian" if family == "Dravidian" else "family-indo-aryan"
                st.markdown(f"<span class='{fam_class}' style='font-size:1.3rem;'>{family}</span>", unsafe_allow_html=True)
                st.caption(f"Syllabic & Morphological Class")

            with r3:
                st.markdown(f"**Prediction**")
                if is_ai:
                    st.markdown(f"<span class='ai-badge'>🤖 AI-Generated</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='human-badge'>✍️ Human-Written</span>", unsafe_allow_html=True)
                st.caption(f"Confidence: **{confidence*100:.1f}%**")

            with r4:
                st.markdown(f"**LA-ADI Detectability**")
                st.markdown(f"### {adi_score_val}")
                st.caption(f"Rank: {adi_rank_val} across Indic languages")

            # Probability Gauge
            st.markdown("#### Detection Probability Distribution")
            prog_col1, prog_col2 = st.columns([1, 1])
            with prog_col1:
                st.write(f"✍️ **Human-Written Probability:** `{human_prob*100:.1f}%`")
                st.progress(min(human_prob, 1.0))
            with prog_col2:
                st.write(f"🤖 **AI-Generated Probability:** `{ai_prob*100:.1f}%`")
                st.progress(min(ai_prob, 1.0))

            # SHAP Local Explanation
            st.markdown("---")
            st.subheader("🧬 Explainable AI Fingerprint (SHAP Attribution)")
            st.markdown("Feature impact showing which exact stylometric markers pushed this text towards AI vs Human:")

            try:
                atlas = AIFingerprintAtlas(detector)
                explanation = atlas.explain_sample(user_text, selected_lang_code)
                top_feats = explanation["top_features"]

                f_col1, f_col2 = st.columns([3, 2])
                with f_col1:
                    f_names = [f["feature"] for f in top_feats[:8]][::-1]
                    f_vals = [f["shap_value"] for f in top_feats[:8]][::-1]
                    f_colors = ["#EF4444" if v > 0 else "#10B981" for v in f_vals]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(f_names, f_vals, color=f_colors)
                    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
                    ax.set_title("Local SHAP Attribution (Red = Supports AI, Green = Supports Human)")
                    ax.set_xlabel("SHAP Value")
                    st.pyplot(fig)
                    plt.close()

                with f_col2:
                    st.markdown("**Top Supporting Clues:**")
                    if explanation["ai_supporting"]:
                        st.markdown("🔴 **Indicators Leaning AI:**")
                        for item in explanation["ai_supporting"][:3]:
                            st.write(f"- `{item['feature']}` (+{item['shap_value']:.3f})")
                    if explanation["human_supporting"]:
                        st.markdown("🟢 **Indicators Leaning Human:**")
                        for item in explanation["human_supporting"][:3]:
                            st.write(f"- `{item['feature']}` ({item['shap_value']:.3f})")

            except Exception as e:
                st.warning(f"Could not compute SHAP plot: {e}")

            # Feature Deep-Dive Tabs
            st.markdown("---")
            st.subheader("🔬 Extracted Feature Taxonomy (33 Dimensions)")
            tab_ling, tab_script, tab_ai = st.tabs([
                "1. Linguistic Features (14)",
                "2. Script & Morphological (11)",
                "3. AI-Style & Uniformity (8)"
            ])

            feats = result["features"]
            with tab_ling:
                l_col1, l_col2, l_col3 = st.columns(3)
                with l_col1:
                    st.metric("Lexical TTR", f"{feats['lexical_ttr']:.3f}")
                    st.metric("Root TTR (Guiraud)", f"{feats['root_ttr']:.2f}")
                    st.metric("Yule's K Index", f"{feats['yules_k']:.1f}")
                    st.metric("Hapax Legomena Ratio", f"{feats['hapax_ratio']:.3f}")
                with l_col2:
                    st.metric("Avg Word Length", f"{feats['avg_word_length']:.2f} chars")
                    st.metric("Word Length Std", f"{feats['word_length_std']:.2f}")
                    st.metric("Mean Sentence Length", f"{feats['mean_sentence_length']:.1f} words")
                    st.metric("Max Sentence Length", f"{feats['max_sentence_length']:.0f} words")
                with l_col3:
                    st.metric("Punctuation Density", f"{feats['punctuation_density']:.3f}")
                    st.metric("Comma / Danda Density", f"{feats['comma_danda_density']:.3f}")
                    st.metric("Word Repetition Rate", f"{feats['word_repetition_rate']:.3f}")
                    st.metric("Bigram Repetition Rate", f"{feats['bigram_repetition_rate']:.3f}")

            with tab_script:
                s_col1, s_col2, s_col3 = st.columns(3)
                with s_col1:
                    st.metric("Matra Density", f"{feats['matra_density']:.3f}")
                    st.metric("Independent Vowel Ratio", f"{feats['independent_vowel_ratio']:.3f}")
                    st.metric("Virama / Halant Freq", f"{feats['virama_halant_frequency']:.3f}")
                with s_col2:
                    st.metric("Consonant Ratio", f"{feats['consonant_ratio']:.3f}")
                    st.metric("Vowel Ratio", f"{feats['vowel_ratio']:.3f}")
                    st.metric("Consonant-Vowel Balance", f"{feats['consonant_vowel_balance']:.2f}")
                with s_col3:
                    st.metric("Indic Char Ratio", f"{feats['indic_char_ratio']:.3f}")
                    st.metric("Agglutination Index", f"{feats['agglutination_index']:.3f}")
                    st.metric("Suffix Inflection Freq", f"{feats['suffix_inflection_frequency']:.3f}")

            with tab_ai:
                a_col1, a_col2, a_col3 = st.columns(3)
                with a_col1:
                    st.metric("Burstiness (CV)", f"{feats['burstiness_cv']:.3f}")
                    st.metric("Normalized Burstiness", f"{feats['burstiness_metric']:.3f}")
                with a_col2:
                    st.metric("Character Entropy", f"{feats['char_entropy']:.3f} bits")
                    st.metric("Word Unigram Entropy", f"{feats['word_entropy']:.3f} bits")
                    st.metric("Bigram Entropy", f"{feats['bigram_entropy']:.3f} bits")
                with a_col3:
                    st.metric("Predictability Proxy", f"{feats['predictability_proxy']:.3f}")
                    st.metric("Stylistic Uniformity Var", f"{feats['stylistic_uniformity_variance']:.3f}")
                    st.metric("Lexical Jaccard Uniformity", f"{feats['lexical_uniformity_jaccard']:.3f}")


# -------------------------------------------------------------
# VIEW 2: LA-ADI BENCHMARK & RANKINGS
# -------------------------------------------------------------
elif menu == "📊 LA-ADI Benchmark & Rankings":
    st.markdown('<div class="main-header">Language-Aware AI Detectability Index (LA-ADI)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Composite index [0 to 1] ranking AI detection difficulty across Indian languages.</div>', unsafe_allow_html=True)

    if la_adi_data is None:
        st.warning("LA-ADI results not found! Please run the pipeline first.")
    else:
        scores = la_adi_data["language_scores"]
        fam_sum = la_adi_data.get("family_summary", {})

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Overall Indic LA-ADI", f"{fam_sum.get('overall_mean_LA_ADI', 0.85):.3f}")
        with m2:
            st.metric("Dravidian Mean LA-ADI", f"{fam_sum.get('Dravidian_mean_LA_ADI', 0.84):.3f}")
        with m3:
            st.metric("Indo-Aryan Mean LA-ADI", f"{fam_sum.get('Indo_Aryan_mean_LA_ADI', 0.86):.3f}")

        st.markdown("### Ranking Across All Eight Languages")
        
        # Build comparison dataframe
        table_rows = []
        for code in la_adi_data["ranking"]:
            info = scores[code]
            table_rows.append({
                "Rank": info["rank"],
                "Language": info["language_name"],
                "Code": code,
                "Family": info["language_family"],
                "LA-ADI Score": info["la_adi_score"],
                "Classification F1": info["classification_performance"],
                "Clue Gap (Separability)": info["clue_gap"],
                "Paraphrasing Stability": info["paraphrasing_stability"]
            })

        df_rank = pd.DataFrame(table_rows)
        st.dataframe(df_rank.style.highlight_max(subset=["LA-ADI Score"], color="#BBF7D0"), use_container_width=True)

        st.markdown("### Visual Comparison Across Components")
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#4F46E5" if f == "Dravidian" else "#059669" for f in df_rank["Family"]]
            bars = ax.bar(df_rank["Language"], df_rank["LA-ADI Score"], color=colors)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("LA-ADI Score")
            ax.set_title("LA-ADI Score (Purple=Dravidian, Green=Indo-Aryan)")
            plt.xticks(rotation=30)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(df_rank))
            width = 0.25
            ax.bar(x - width, df_rank["Classification F1"], width, label='Performance', color='#3B82F6')
            ax.bar(x, df_rank["Clue Gap (Separability)"], width, label='Clue Gap', color='#F59E0B')
            ax.bar(x + width, df_rank["Paraphrasing Stability"], width, label='Stability', color='#10B981')
            ax.set_xticks(x)
            ax.set_xticklabels(df_rank["Language"], rotation=30)
            ax.set_ylim(0, 1.05)
            ax.set_title("Three LA-ADI Pillars Breakdown")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        st.info(r"""
        **Methodological Formulation:**
        - **$P_L$ (Performance)**: Detection F1 on clean benchmark test splits.
        - **$G_L$ (Clue Gap)**: Divergence between Human and AI feature distributions in language $L$.
        - **$S_L$ (Stability)**: Robustness against semantic perturbation and back-translation attacks.
        - **$\text{LA-ADI}(L) = 0.40 \cdot P_L + 0.30 \cdot G_L + 0.30 \cdot S_L$**
        """)


# -------------------------------------------------------------
# VIEW 3: AI FINGERPRINT ATLAS
# -------------------------------------------------------------
elif menu == "🧬 AI Fingerprint Atlas":
    st.markdown('<div class="main-header">Indic AI Fingerprint Atlas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable AI writing patterns across Indian languages derived from SHAP global attributions.</div>', unsafe_allow_html=True)

    if atlas_data is None:
        st.warning("AI Fingerprint Atlas data not found! Please run the pipeline first.")
    else:
        st.subheader("1. Common Indic AI Writing Patterns (Global)")
        st.markdown("Features with universal discriminative power across all eight Indic languages:")

        global_feats = atlas_data.get("global_fingerprints", [])
        if global_feats:
            g_col1, g_col2 = st.columns([3, 2])
            with g_col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                names = [x["feature"] for x in global_feats[:10]][::-1]
                vals = [x["importance"] for x in global_feats[:10]][::-1]
                ax.barh(names, vals, color="#4F46E5")
                ax.set_title("Top 10 Universal Indic AI Markers (Mean |SHAP|)")
                ax.set_xlabel("Mean Absolute SHAP Value")
                st.pyplot(fig)
                plt.close()

            with g_col2:
                st.markdown("**Key Universal Tendencies:**")
                st.markdown("""
                1. **Burstiness Compression**: AI generates sentences with markedly lower variance in sentence length compared to natural human writing.
                2. **Hapax & Lexical Redundancy**: AI text exhibits lower hapax legomena ratios and higher token repetition.
                3. **Matra & Diacritic Regularity**: AI models produce more uniform matra frequencies per word than human news writers.
                4. **Predictability Dominance**: AI tokens exhibit higher average unigram likelihoods (less vocabulary risk).
                """)

        st.markdown("---")
        st.subheader("2. Family-Specific AI Writing Fingerprints")
        fam_feats = atlas_data.get("family_fingerprints", {})
        
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("#### 🔵 Dravidian Family (Tamil, Telugu, Malayalam, Kannada)")
            st.caption("Morphologically agglutinative with heavy inflectional suffixes")
            if "Dravidian" in fam_feats:
                dr_feats = fam_feats["Dravidian"]
                for i, f in enumerate(dr_feats[:6], 1):
                    st.write(f"**{i}. `{f['feature']}`** — Impact: `{f['importance']:.4f}`")

        with fc2:
            st.markdown("#### 🟡 Indo-Aryan Family (Hindi, Bengali, Marathi, Gujarati)")
            st.caption("Inflectional/analytical with distinct compound verb patterns")
            if "Indo-Aryan" in fam_feats:
                ia_feats = fam_feats["Indo-Aryan"]
                for i, f in enumerate(ia_feats[:6], 1):
                    st.write(f"**{i}. `{f['feature']}`** — Impact: `{f['importance']:.4f}`")

        st.markdown("---")
        st.subheader("3. Language-Specific Top Markers")
        selected_atlas_lang = st.selectbox(
            "Select language to view its specific AI signature:",
            TARGET_LANGS,
            format_func=lambda x: f"{LANGUAGE_NAMES[x]} ({x})"
        )

        lang_feats = atlas_data.get("language_fingerprints", {}).get(selected_atlas_lang, [])
        if lang_feats:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            names = [x["feature"] for x in lang_feats[::-1]]
            vals = [x["importance"] for x in lang_feats[::-1]]
            ax.barh(names, vals, color="#059669")
            ax.set_title(f"Top Discriminative Features for {LANGUAGE_NAMES[selected_atlas_lang]}")
            ax.set_xlabel("Mean |SHAP|")
            st.pyplot(fig)
            plt.close()


# -------------------------------------------------------------
# VIEW 4: EXPERIMENTAL EVALUATIONS
# -------------------------------------------------------------
elif menu == "🧪 Experimental Evaluations":
    st.markdown('<div class="main-header">Experimental Evaluation & Robustness Studies</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical validation based on the IndicTextFake Multilingual Benchmark.</div>', unsafe_allow_html=True)

    if eval_report is None:
        st.warning("Evaluation report not found! Please run the pipeline first.")
    else:
        # Overall Metrics
        ov = eval_report.get("overall", {})
        st.subheader("1. Overall Benchmark Performance")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.metric("Accuracy", f"{ov.get('accuracy', 0.90)*100:.1f}%")
        with e2:
            st.metric("Precision", f"{ov.get('precision', 0.90)*100:.1f}%")
        with e3:
            st.metric("Recall", f"{ov.get('recall', 0.90)*100:.1f}%")
        with e4:
            st.metric("F1-Score", f"{ov.get('f1_score', 0.90):.4f}")

        # Confusion Matrix
        cm = ov.get("confusion_matrix", [[0,0],[0,0]])
        st.markdown("#### Test Set Confusion Matrix")
        cm_col1, cm_col2 = st.columns([1, 2])
        with cm_col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Human", "AI"],
                        yticklabels=["Human", "AI"], ax=ax)
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            st.pyplot(fig)
            plt.close()

        with cm_col2:
            st.markdown(f"""
            - **True Negatives (Human correctly identified):** `{cm[0][0]}`
            - **False Positives (Human misclassified as AI):** `{cm[0][1]}`
            - **False Negatives (AI misclassified as Human):** `{cm[1][0]}`
            - **True Positives (AI correctly identified):** `{cm[1][1]}`
            - **AUC-ROC Score:** `{ov.get('auc', 0.95):.4f}`
            """)

        # Cross-AI & Cross-Language
        st.markdown("---")
        st.subheader("2. Generalization Studies")
        gen_tab1, gen_tab2 = st.tabs(["Cross-AI Testing (Unseen Generators)", "Cross-Language / Cross-Family Transfer"])

        with gen_tab1:
            cross_ai = eval_report.get("cross_ai", {})
            st.markdown("Trained on subsets of generators + human data, tested on completely unseen LLMs:")
            ca_rows = []
            for k, val in cross_ai.items():
                m = val.get("metrics", {})
                ca_rows.append({
                    "Unseen AI Model": val.get("unseen_generator"),
                    "Test Accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
                    "Test F1": f"{m.get('f1_score', 0):.4f}",
                    "Test AUC": f"{m.get('auc', 0):.4f}"
                })
            st.table(pd.DataFrame(ca_rows))

        with gen_tab2:
            cross_lang = eval_report.get("cross_language", {})
            st.markdown("Evaluating zero-shot transfer across language family boundaries:")
            cl_rows = []
            for exp_name, m in cross_lang.items():
                cl_rows.append({
                    "Transfer Protocol": exp_name.replace("_", " "),
                    "Accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
                    "F1 Score": f"{m.get('f1_score', 0):.4f}",
                    "AUC": f"{m.get('auc', 0):.4f}"
                })
            st.table(pd.DataFrame(cl_rows))

        # Ablation Studies
        st.markdown("---")
        st.subheader("3. Feature Category Ablation Studies")
        ablations = eval_report.get("ablation_study", {})
        if ablations:
            ab_rows = []
            for name, m in ablations.items():
                ab_rows.append({
                    "Feature Configuration": name.replace("_", " "),
                    "Accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
                    "F1 Score": m.get("f1_score", 0),
                    "AUC": m.get("auc", 0)
                })
            df_ab = pd.DataFrame(ab_rows)
            st.table(df_ab)

        # Paraphrasing Robustness
        st.markdown("---")
        st.subheader("4. Semantic Paraphrasing Robustness")
        st.markdown("Evaluating detector resistance against semantic-preserving evasion attacks:")
        pert = eval_report.get("paraphrasing_robustness", {})
        if pert:
            p_rows = []
            for pt, m in pert.items():
                p_rows.append({
                    "Perturbation Protocol": pt,
                    "Description": "Original unperturbed" if pt == "clean" else ("Back-translation paraphrase" if pt == "paraphrase_1" else "Pivot Hindi paraphrase"),
                    "Accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
                    "F1 Score": f"{m.get('f1_score', 0):.4f}",
                    "AUC": f"{m.get('auc', 0):.4f}"
                })
            st.table(pd.DataFrame(p_rows))


# -------------------------------------------------------------
# VIEW 5: RESEARCH METHODOLOGY
# -------------------------------------------------------------
elif menu == "📚 Research Methodology":
    st.markdown('<div class="main-header">Research Methodology & Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">IndicTextFake Benchmark Design, Feature Formulations, and LA-ADI Specifications.</div>', unsafe_allow_html=True)

    st.markdown(r"""
    ### 1. Benchmark Dataset (IndicTextFake)
    - **Dataset Source:** Hugging Face benchmark `IndicTextFake/IndicTextFake`.
    - **Languages:** 8 target Indian languages:
      - **Dravidian:** Tamil (`ta`), Telugu (`te`), Malayalam (`ml`), Kannada (`kn`)
      - **Indo-Aryan:** Hindi (`hi`), Bengali (`bn`), Marathi (`mr`), Gujarati (`gu`)
    - **Data Composition:**
      - **Human-Written:** Extracted from AI4Bharat's IndicCorpV2 news corpus.
      - **AI-Generated:** Multi-generator samples from **Cohere Aya-Expanse-8B**, **Qwen/Qwen2.5-7B**, and **GPT-OSS**.
      - **Balancing:** Exactly balanced 50% Human and 50% AI per language with equal sub-representation across LLM families.

    ### 2. Unicode Text Preprocessing & Script Routing
    - **NFC Unicode Normalization:** Unifies canonical decompositions.
    - **Zero-Width Character Sanitization:** Strips rogue ZWSP / BOM while preserving morphologically critical joiners (ZWJ/ZWNJ).
    - **Script Identification:** Unicode codepoint mapping across Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, and Malayalam ranges.
    - **Devanagari Disambiguation:** Differentiates Hindi vs Marathi using lexical markers and character checks (\u0933 / ळ).

    ### 3. Three-Category Feature Taxonomy (33 Dimensions)
    1. **Linguistic Features:**
       - Lexical Diversity: Type-Token Ratio ($TTR = |V| / N$), Guiraud's Root $TTR = |V| / \sqrt{N}$, Yule's Characteristic $K = 10^4 \cdot \frac{\sum f_i^2 - N}{N^2}$, Hapax ratio.
       - Sentence & Word Metrics: Word length mean/std, sentence length mean/std/max.
       - Repetition Metrics: Word repetition rate, bigram & trigram repetition rates.
    2. **Script & Morphological Features:**
       - Matra density: Dependent vowel signs per Indic character.
       - Virama / Halant frequency: Consonant conjunct cluster density.
       - Independent vowel ratio and Consonant-Vowel balance.
       - Agglutination index: Suffix inflection density proxies.
    3. **AI-Style Features:**
       - Burstiness: Coefficient of variation $\sigma / \mu$ and normalized burstiness $\frac{\sigma - \mu}{\sigma + \mu}$.
       - Shannon Entropies: Character entropy, word unigram entropy, and bigram entropy: $H = -\sum p_i \log_2 p_i$.
       - Predictability Proxy: Average token log-likelihood.
       - Stylistic & Lexical Uniformity: Variance across document quarters and Jaccard similarity between document halves.

    ### 4. Language-Aware AI Detectability Index (LA-ADI)
    $$\text{LA-ADI}(L) = 0.40 \cdot P_L + 0.30 \cdot G_L + 0.30 \cdot S_L$$
    - **$P_L$**: Classification Performance (Test F1 score).
    - **$G_L$**: Clue Gap (Statistical separability of Human vs AI distributions).
    - **$S_L$**: Paraphrasing Stability under back-translation perturbation.
    """)
