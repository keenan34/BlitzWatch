import { useState } from "react";

export default function BlitzWatch() {
  const [form, setForm] = useState({
    down: 1,
    ydstogo: 10,
    yardline_100: 50,
    qtr: 1,
    min_left: 15,
    sec_left: 0,
    posteam_score: 0,
    defteam_score: 0,
    pass_location: "middle",
    pass_length: "short",
    shotgun: false,
    no_huddle: false,
  });
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((f) => ({
      ...f,
      [name]: type === "checkbox" ? checked : Number(value) || value,
    }));
  };

  const handleSubmit = async () => {
    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      setResult(await res.json());
    } catch {
      alert("⚠️ Cannot reach backend. Is Flask running on port 5000?");
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-8 p-6 space-y-8">
      <h1 className="text-4xl font-bold text-center">BlitzWatch Predictor</h1>

      {/* Form Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        {[
          { name: "down", label: "Down", placeholder: "1–4" },
          { name: "ydstogo", label: "Yards to Go", placeholder: "1–99" },
          { name: "yardline_100", label: "Yardline to Endzone", placeholder: "1–99" },
          { name: "qtr", label: "Quarter", placeholder: "1–4" },
          { name: "min_left", label: "Min Left", placeholder: "0–15" },
          { name: "sec_left", label: "Sec Left", placeholder: "0–59" },
          { name: "posteam_score", label: "Your Score", placeholder: "0–100" },
          { name: "defteam_score", label: "Opp Score", placeholder: "0–100" },
        ].map(({ name, label, placeholder }) => (
          <div key={name} className="flex flex-col">
            <label htmlFor={name} className="mb-1 font-medium">
              {label}
            </label>
            <input
              id={name}
              name={name}
              type="number"
              min={0}
              value={form[name]}
              onChange={handleChange}
              className="p-2 border rounded focus:ring focus:ring-blue-200"
              placeholder={placeholder}
            />
          </div>
        ))}

        {/* Pass options */}
        <div className="flex flex-col">
          <label htmlFor="pass_location" className="mb-1 font-medium">
            Pass Location
          </label>
          <select
            id="pass_location"
            name="pass_location"
            value={form.pass_location}
            onChange={handleChange}
            className="p-2 border rounded focus:ring focus:ring-blue-200"
          >
            <option value="left">Left</option>
            <option value="middle">Middle</option>
            <option value="right">Right</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label htmlFor="pass_length" className="mb-1 font-medium">
            Pass Length
          </label>
          <select
            id="pass_length"
            name="pass_length"
            value={form.pass_length}
            onChange={handleChange}
            className="p-2 border rounded focus:ring focus:ring-blue-200"
          >
            <option value="short">Short</option>
            <option value="deep">Deep</option>
            <option value="none">None</option>
          </select>
        </div>

        {/* Toggles */}
        <div className="flex items-center space-x-2">
          <input
            id="shotgun"
            type="checkbox"
            name="shotgun"
            checked={form.shotgun}
            onChange={handleChange}
            className="h-5 w-5"
          />
          <label htmlFor="shotgun" className="font-medium">
            Shotgun
          </label>
        </div>
        <div className="flex items-center space-x-2">
          <input
            id="no_huddle"
            type="checkbox"
            name="no_huddle"
            checked={form.no_huddle}
            onChange={handleChange}
            className="h-5 w-5"
          />
          <label htmlFor="no_huddle" className="font-medium">
            No Huddle
          </label>
        </div>
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        className="w-full py-3 bg-blue-600 text-white text-lg font-semibold rounded hover:bg-blue-700 transition"
      >
        Predict Blitz
      </button>

      {/* Result Display */}
      {result && (
        <div className="p-4 bg-gray-100 rounded shadow">
          <p>
            Blitz Probability: <strong>{(result.proba * 100).toFixed(2)}%</strong>
          </p>
          <p
            className={`mt-2 text-lg font-semibold ${
              result.proba > 0.5 ? "text-red-600" : "text-green-600"
            }`}
          >
            → {result.proba > 0.5 ? "BLITZ LIKELY" : "NO BLITZ"}
          </p>
        </div>
      )}

      {/* Model Insights */}
      <div className="mt-8 space-y-6">
        <h2 className="text-2xl font-bold">Model Insights</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-medium mb-2">Feature Importance</h3>
            <img
              src="http://localhost:5000/insights/feature_importance"
              alt="Feature Importance"
              className="w-full rounded shadow"
            />
          </div>
          <div>
            <h3 className="font-medium mb-2">SHAP Summary</h3>
            <img
              src="http://localhost:5000/insights/shap_summary"
              alt="SHAP Summary"
              className="w-full rounded shadow"
            />
          </div>
          <div>
            <h3 className="font-medium mb-2">Confusion Matrix</h3>
            <img
              src="http://localhost:5000/insights/confusion_matrix"
              alt="Confusion Matrix"
              className="w-full rounded shadow"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
