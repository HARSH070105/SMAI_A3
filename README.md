# SMAI_A3 | T12.8: Monument Scavenger Hunt Game 
### The Problem
Tourists often want to know more about the historical monuments they are visiting, but traditional searches take them out of the moment. Furthermore, exploring heritage sites can sometimes lack an engaging, interactive element.

### What We Built
- This project is a gamified Streamlit web application that transforms a standard monument identifier into a scavenger hunt.

- The app generates cryptic clues about a specific Indian monument.

- The user figures out the clue and uploads a photograph of the correct monument.

- Using an AI vision model, the app verifies if the user found the right place.

- Upon success, it rewards the user with a historical paragraph, a visit info card (opening hours and ticket prices), and an "Open in Google Maps" link.

### Features
- **Gamified Experience:** Users are challenged with clues rather than just identifying random images.

- **Zero-Shot Image Classification:** Utilizes OpenAI's clip-vit-base-patch32 to instantly match user-uploaded photos to a class-name list without requiring heavy model re-training.

- **Rich Metadata Integration:** Displays cached JSON data (history, hours, ticket prices) scraped from Wikipedia or generated via Gemini.

- **Seamless Navigation:** Direct integration with Google Maps to help users locate the monument.

### Tech Stack & Skills
Frontend: Streamlit

Machine Learning: Zero-shot classification using openai/clip-vit-base-patch32 (via Hugging Face Transformers).

Data Sources: * Images: Indian Monuments Image Dataset by Danushkumarv (24 monuments, ~3.5k images) & Wikimedia Commons.

Metadata: Pre-cached JSON containing historical facts, timings, and ticket prices.
