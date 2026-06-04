


import React, { useState, useRef } from "react";
const diseaseSolutions = {
  "Bacterial Leaf Blight": 
    "• Avoid water stagnation (खेत में पानी जमा न होने दें)\n" +
    "• Use balanced fertilizers, avoid excess nitrogen (संतुलित खाद दें, ज्यादा यूरिया न डालें)\n" +
    "• Spray copper-based bactericide (कॉपर आधारित दवा का छिड़काव करें)\n" +
    "• Remove infected leaves (रोगी पत्तों को हटा दें)",

  "Brown Spot": 
    "• Improve soil fertility (मिट्टी की उर्वरता बढ़ाएं)\n" +
    "• Maintain proper irrigation (पानी की सही मात्रा रखें)\n" +
    "• Spray Mancozeb fungicide (Mancozeb दवा का छिड़काव करें)\n" +
    "• Provide proper nutrition (पौधों को पोषण दें)",

  "Leaf Blast": 
    "• Spray Tricyclazole fungicide (Tricyclazole दवा का छिड़काव करें)\n" +
    "• Maintain plant spacing (पौधों के बीच दूरी रखें)\n" +
    "• Avoid excess moisture (ज्यादा नमी से बचें)\n" +
    "• Remove infected leaves (रोगी पत्तों को हटा दें)",

  "Rice Leaf Blast": 
    "• Spray Tricyclazole fungicide (Tricyclazole दवा का छिड़काव करें)\n" +
    "• Keep proper water level (पानी संतुलित रखें)\n" +
    "• Avoid overcrowding (पौधों को ज्यादा घना न रखें)\n" +
    "• Destroy infected leaves (रोगी पत्तों को नष्ट करें)",

  "Healthy": 
    "• Crop is healthy (फसल स्वस्थ है ✅)\n" +
    "• Maintain regular irrigation (नियमित पानी दें)\n" +
    "• Apply balanced fertilizers (संतुलित खाद दें)\n" +
    "• Monitor regularly (समय-समय पर जांच करें)"
};
const getSolution = (disease) => {
  if (!disease) return "No solution available";

  if (disease.includes("Blast"))
    return diseaseSolutions["Leaf Blast"];

  if (disease.includes("Brown"))
    return diseaseSolutions["Brown Spot"];

  if (disease.includes("Bacterial"))
    return diseaseSolutions["Bacterial Leaf Blight"];

  if (disease.includes("Healthy"))
    return diseaseSolutions["Healthy"];

  return "No solution available";
};
import { Upload, Image as ImageIcon, Check, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";





const ImageUploadSection = () => {
  const [image, setImage] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const { toast } = useToast();

  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
    }
  };

  const handleImageFile = (file) => {
    if (!file.type.match("image.*")) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (JPEG, PNG, etc.)",
        variant: "destructive",
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        setImage(e.target.result);
        setResult(null);
      }
    };
    reader.readAsDataURL(file);
  };

  const analyzeImage = async () => {
    if (!image) return;

    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();

      // Convert base64 image (from FileReader) back to Blob
      const res = await fetch(image);
      const blob = await res.blob();
      formData.append("file", blob, "leaf-image.jpg");

      const response = await fetch("https://rice-leaf-disease-detection-waco.onrender.com/predict"), {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.prediction) {
        // setResult(data.prediction);
        const cleanResult = data.prediction
           .replace(/_/g, " ")
           .replace(/\s+/g, " ")
            .trim();

           setResult(cleanResult);

        toast({
          title: "Analysis Complete",
          description: `Prediction: ${data.prediction}`,
        });
      } else {
        throw new Error("No prediction received");
      }
    } catch (error) {
      console.error("Error analyzing image:", error);

      toast({
        title: "Error",
        description: "Something went wrong during image analysis.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section
      id="try-now"
      className="py-16 bg-gradient-to-b from-white to-gray-50 px-4"
    >
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Try PaddyGuard Now
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload a photo of your plant's leaves to get an instant diagnosis.
            It's free and no registration required.
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 md:p-8">
          {!image ? (
            <div
              className={`leaf-image-upload border-2 border-dashed rounded-lg p-6 text-center transition-all ${
                isDragging ? "border-leaf bg-leaf/5" : "border-gray-300"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleFileInput}
              />

              <Upload className="w-16 h-16 text-leaf mb-4 mx-auto" />
              <h3 className="text-xl font-semibold mb-2">
                Upload Your Leaf Image
              </h3>
              <p className="text-gray-500 mb-4">
                Drag and drop an image here, or click the button below
              </p>

              <Button
                type="button"
                className="bg-leaf hover:bg-leaf-dark"
                onClick={() => fileInputRef.current?.click()}
              >
                <ImageIcon className="mr-2 h-4 w-4" /> Select Image
              </Button>
            </div>
          ) : (
            <div className="space-y-8">
              <div className="relative rounded-lg overflow-hidden">
                <img
                  src={image}
                  alt="Uploaded leaf"
                  className="w-full h-auto"
                />
                <Button
                  variant="outline"
                  size="sm"
                  className="absolute top-4 right-4"
                  onClick={() => setImage(null)}
                >
                  Change Image
                </Button>
              </div>

              {!result ? (
                <div className="flex justify-center">
                  <Button
                    size="lg"
                    className="bg-leaf hover:bg-leaf-dark"
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        Analyzing...
                        <div className="ml-2 animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                      </>
                    ) : (
                      <>Analyze Image</>
                    )}
                  </Button>
                </div>
              ) : (
                <div className="bg-leaf/10 border border-leaf p-6 rounded-lg">
                  <div className="flex items-start space-x-4">
                    <Check className="w-6 h-6 text-leaf mt-1 flex-shrink-0" />
                    <div>
                      <h3 className="text-xl font-semibold mb-2">
                        Diagnosis Result
                      </h3>
                      {/* <p className="text-gray-700">{result}</p> */}
                      <p className="text-gray-700">
                     <strong>Disease:</strong> {result}
                   </p>

                      <p className="text-gray-700 mt-2">
                      {/* <strong>Solution:</strong> {diseaseSolutions[result] || "No solution available"} */}
                      <strong>Solution:</strong> {getSolution(result)}
                   </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="mt-10 flex justify-center">
          <div className="flex items-center text-sm text-gray-500">
            <AlertCircle className="w-4 h-4 mr-2" />
            <p>
              For demonstration purposes only. In a real app, analysis would be
              performed by our AI.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ImageUploadSection;
