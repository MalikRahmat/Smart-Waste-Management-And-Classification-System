def get_recommendation(label):
    data = {
        "plastic": "Place in Recyclable Plastic Bin.",
        "glass": "Dispose in Glass Recycling Container.",
        "organic": "Compost this material.",
        "paper": "Put in Paper Recycling Bin.",
        "metal": "Send to Metal Recycling Facility.",
        "other": "Dispose according to local waste guidelines."

    }
    return data.get(label, "No recommendation available.")
