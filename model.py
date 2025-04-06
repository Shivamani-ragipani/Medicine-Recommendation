import numpy as np
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MedicineRecommender:
    def __init__(self):
        # Initialize the symptom database
        self.symptom_db = self._create_symptom_database()
        
        # Initialize the TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Fit the vectorizer on our symptom keywords
        all_keywords = []
        for condition in self.symptom_db:
            all_keywords.extend(condition['keywords'])
        self.vectorizer.fit_transform(all_keywords)
    
    def _create_symptom_database(self):
        """
        Create a database of symptoms, conditions, and recommended medicines.
        In a production environment, this would be loaded from a proper database.
        """
        return [
            {
                'id': 1,
                'condition': 'Headache',
                'keywords': ['headache', 'head pain', 'migraine', 'head ache', 'pounding head', 
                             'throbbing head', 'tension headache', 'pain in head', 'head hurts'],
                'medicines': [
                    {
                        'name': 'Paracetamol (Acetaminophen)',
                        'dosage': '500-1000mg every 4-6 hours as needed (max 4g per day)',
                        'description': 'A pain reliever and fever reducer that works by blocking the production of certain natural substances that cause pain and fever.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a681004.html'
                    },
                    {
                        'name': 'Ibuprofen',
                        'dosage': '200-400mg every 4-6 hours as needed (max 1200mg per day)',
                        'description': 'A nonsteroidal anti-inflammatory drug (NSAID) that reduces hormones that cause inflammation and pain in the body.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682159.html'
                    },
                    {
                        'name': 'Aspirin',
                        'dosage': '325-650mg every 4-6 hours as needed',
                        'description': 'Reduces pain, inflammation, and fever. Not recommended for children under 12 years of age.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682878.html'
                    }
                ],
                'recommendation': 'For headaches, rest in a quiet, dark room and stay hydrated. If symptoms persist for more than 3 days or are severe, please consult a doctor.',
                'is_critical': False
            },
            {
                'id': 2,
                'condition': 'Fever',
                'keywords': ['fever', 'high temperature', 'hot', 'feeling hot', 'temperature', 
                             'chills', 'sweating', 'body heat', 'feverish', 'elevated temperature'],
                'medicines': [
                    {
                        'name': 'Paracetamol (Acetaminophen)',
                        'dosage': '500-1000mg every 4-6 hours as needed (max 4g per day)',
                        'description': 'Helps reduce fever by affecting the heat-regulating center of the brain.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a681004.html'
                    },
                    {
                        'name': 'Ibuprofen',
                        'dosage': '200-400mg every 4-6 hours as needed (max 1200mg per day)',
                        'description': 'Reduces fever and inflammation by blocking the body\'s production of certain natural substances.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682159.html'
                    }
                ],
                'recommendation': 'For fever, ensure adequate rest and hydration. If fever persists for more than 3 days or exceeds 39°C (102.2°F), please consult a doctor.',
                'is_critical': False
            },
            {
                'id': 3,
                'condition': 'Common Cold',
                'keywords': ['cold', 'runny nose', 'congestion', 'stuffy nose', 'sneezing', 
                             'sore throat', 'cough', 'nasal congestion', 'blocked nose', 'mucus'],
                'medicines': [
                    {
                        'name': 'Pseudoephedrine',
                        'dosage': '60mg every 4-6 hours (max 240mg per day)',
                        'description': 'A decongestant that shrinks blood vessels in the nasal passages to reduce stuffiness.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682619.html'
                    },
                    {
                        'name': 'Cetirizine',
                        'dosage': '10mg once daily',
                        'description': 'An antihistamine that reduces the effects of natural chemical histamine in the body, which can produce symptoms of sneezing and runny nose.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a698026.html'
                    },
                    {
                        'name': 'Dextromethorphan',
                        'dosage': '10-20mg every 4 hours or 30mg every 6-8 hours',
                        'description': 'A cough suppressant that works by decreasing activity in the part of the brain that causes coughing.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682492.html'
                    }
                ],
                'recommendation': 'For cold symptoms, rest and stay hydrated. Over-the-counter medicines can help manage symptoms. If symptoms worsen or last more than 10 days, please consult a doctor.',
                'is_critical': False
            },
            {
                'id': 4,
                'condition': 'Stomach Pain',
                'keywords': ['stomach pain', 'abdominal pain', 'stomach ache', 'indigestion', 'heartburn',
                             'acid reflux', 'upset stomach', 'stomach cramps', 'bloating', 'gas'],
                'medicines': [
                    {
                        'name': 'Omeprazole',
                        'dosage': '20mg once daily before a meal',
                        'description': 'A proton pump inhibitor that decreases the amount of acid produced in the stomach.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a693050.html'
                    },
                    {
                        'name': 'Antacid (Calcium Carbonate)',
                        'dosage': '1-2 tablets as needed',
                        'description': 'Neutralizes stomach acid to relieve heartburn, acid indigestion, and upset stomach.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a601032.html'
                    },
                    {
                        'name': 'Simethicone',
                        'dosage': '40-125mg after meals and at bedtime',
                        'description': 'Helps break up gas bubbles in the gut to relieve bloating and discomfort.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682683.html'
                    }
                ],
                'recommendation': 'For mild stomach discomfort, try over-the-counter remedies. Avoid spicy or fatty foods. If pain is severe, persistent, or accompanied by other symptoms, please consult a doctor.',
                'is_critical': False
            },
            {
                'id': 5,
                'condition': 'Joint and Muscle Pain',
                'keywords': ['knee pain', 'joint pain', 'arthritis', 'muscle pain', 'backache', 
                             'back pain', 'muscle ache', 'stiffness', 'sore muscles', 'joint stiffness'],
                'medicines': [
                    {
                        'name': 'Ibuprofen',
                        'dosage': '200-400mg every 4-6 hours as needed (max 1200mg per day)',
                        'description': 'Reduces inflammation and pain in joints and muscles.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682159.html'
                    },
                    {
                        'name': 'Diclofenac Gel',
                        'dosage': 'Apply to affected area 4 times daily',
                        'description': 'A topical NSAID that reduces pain and inflammation when applied directly to the affected area.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a611002.html'
                    },
                    {
                        'name': 'Naproxen',
                        'dosage': '220mg every 8-12 hours (max 660mg per day)',
                        'description': 'An NSAID that works by stopping the body\'s production of a substance that causes pain and inflammation.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a681029.html'
                    }
                ],
                'recommendation': 'For joint or muscle pain, rest the affected area and apply ice for 15-20 minutes several times a day. If pain persists for more than a week or is severe, please consult a doctor.',
                'is_critical': False
            },
            {
                'id': 6,
                'condition': 'Allergies',
                'keywords': ['allergy', 'allergic reaction', 'hay fever', 'itchy eyes', 'itchy skin', 
                             'rash', 'hives', 'allergies', 'sneezing', 'itching'],
                'medicines': [
                    {
                        'name': 'Cetirizine',
                        'dosage': '10mg once daily',
                        'description': 'An antihistamine that reduces allergy symptoms such as watery eyes, runny nose, itching, and sneezing.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a698026.html'
                    },
                    {
                        'name': 'Loratadine',
                        'dosage': '10mg once daily',
                        'description': 'A non-drowsy antihistamine that relieves allergy symptoms.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a697038.html'
                    },
                    {
                        'name': 'Hydrocortisone Cream',
                        'dosage': 'Apply to affected area 2-4 times daily',
                        'description': 'A topical corticosteroid that reduces the swelling, itching, and redness that can occur in skin conditions.',
                        'url': 'https://medlineplus.gov/druginfo/meds/a682793.html'
                    }
                ],
                'recommendation': 'For allergies, avoid known triggers when possible. Over-the-counter antihistamines can help manage symptoms. If you experience severe allergic reactions or difficulty breathing, seek immediate medical attention.',
                'is_critical': False
            },
            # Critical conditions that require medical attention
            {
                'id': 7,
                'condition': 'Chest Pain',
                'keywords': ['chest pain', 'heart', 'breathing difficulty', 'shortness of breath', 
                             'chest pressure', 'chest tightness', 'heart attack', 'cardiac', 'angina'],
                'medicines': [],
                'recommendation': 'These symptoms may indicate a serious condition such as a heart attack or respiratory issue. Please seek immediate medical attention or call emergency services.',
                'is_critical': True
            },
            {
                'id': 8,
                'condition': 'Severe Headache',
                'keywords': ['severe headache', 'worst headache', 'sudden headache', 'stiff neck', 'confusion',
                             'worst headache of my life', 'thunderclap headache', 'headache with fever and stiff neck'],
                'medicines': [],
                'recommendation': 'These symptoms may indicate a serious neurological condition. Please seek immediate medical attention or call emergency services.',
                'is_critical': True
            },
            {
                'id': 9,
                'condition': 'Unconsciousness',
                'keywords': ['unconscious', 'fainted', 'seizure', 'convulsion', 'passed out', 
                             'loss of consciousness', 'not responsive', 'collapse', 'blackout'],
                'medicines': [],
                'recommendation': 'This is a medical emergency. Please call emergency services immediately.',
                'is_critical': True
            },
            {
                'id': 10,
                'condition': 'High Fever',
                'keywords': ['very high fever', 'fever above 103', 'fever above 39.5', 'extreme fever',
                             'fever with rash', 'persistent high fever', 'fever not responding to medication'],
                'medicines': [],
                'recommendation': 'A very high fever may indicate a serious infection or condition. Please seek immediate medical attention, especially if accompanied by confusion, severe headache, or rash.',
                'is_critical': True
            }
        ]
    
    def recommend(self, symptoms_text, additional_info=None):
        """
        Analyze symptoms and provide medicine recommendations
        
        Args:
            symptoms_text (str): User's description of symptoms
            additional_info (dict): Additional information like duration, severity, etc.
            
        Returns:
            dict: Recommendation results including medicines and advice
        """
        # Convert to lowercase for processing
        symptoms_text = symptoms_text.lower()
        
        # Check for critical conditions first (safety first approach)
        for condition in self.symptom_db:
            if condition['is_critical']:
                # Use more sophisticated matching for critical conditions
                if self._check_critical_match(symptoms_text, condition['keywords']):
                    return {
                        'recommendation': condition['recommendation'],
                        'is_critical': True,
                        'medicines': []
                    }
        
        # For non-critical conditions, use ML-based similarity matching
        matched_conditions = self._find_matching_conditions(symptoms_text)
        
        if not matched_conditions:
            return {
                'recommendation': "I couldn't identify your symptoms clearly. Please provide more details or consult a healthcare professional for proper diagnosis.",
                'is_critical': False,
                'medicines': []
            }
        
        # Combine recommendations and medicines from matched conditions
        all_medicines = []
        recommendations = []
        
        for condition, score in matched_conditions:
            recommendations.append(condition['recommendation'])
            
            # Only include medicines from conditions with good match scores
            if score > 0.2:  # Threshold for including medicines
                for medicine in condition['medicines']:
                    # Check if medicine is already in the list to avoid duplicates
                    if not any(m['name'] == medicine['name'] for m in all_medicines):
                        all_medicines.append(medicine)
        
        # Use additional info to refine recommendations if available
        if additional_info:
            all_medicines = self._refine_with_additional_info(all_medicines, additional_info)
        
        return {
            'recommendation': ' '.join(recommendations),
            'is_critical': False,
            'medicines': all_medicines
        }
    
    def _check_critical_match(self, symptoms_text, keywords):
        """
        More careful matching for critical conditions to avoid false negatives
        """
        # For critical conditions, we want to be more sensitive
        for keyword in keywords:
            # Check for exact matches or phrases
            if re.search(r'\b' + re.escape(keyword) + r'\b', symptoms_text):
                return True
        return False
    
    def _find_matching_conditions(self, symptoms_text):
        """
        Use TF-IDF and cosine similarity to find matching conditions
        """
        # Transform the input text
        symptoms_vector = self.vectorizer.transform([symptoms_text])
        
        matched_conditions = []
        
        for condition in self.symptom_db:
            if not condition['is_critical']:  # Skip critical conditions as they're handled separately
                # Combine keywords for better matching
                condition_text = ' '.join(condition['keywords'])
                condition_vector = self.vectorizer.transform([condition_text])
                
                # Calculate similarity
                similarity = cosine_similarity(symptoms_vector, condition_vector)[0][0]
                
                # If similarity is above threshold, consider it a match
                if similarity > 0.1:  # Low threshold to catch more potential matches
                    matched_conditions.append((condition, similarity))
        
        # Sort by similarity score
        matched_conditions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        return matched_conditions[:3]
    
    def _refine_with_additional_info(self, medicines, additional_info):
        """
        Refine medicine recommendations based on additional information
        """
        refined_medicines = medicines.copy()
        
        # Example: If user has allergies, avoid certain medications
        if additional_info.get('conditions') and 'Allergies' in additional_info['conditions']:
            # Filter out medicines that might cause allergic reactions
            refined_medicines = [m for m in refined_medicines if 'aspirin' not in m['name'].lower()]
        
        # Example: If symptoms are severe, prioritize stronger medications
        if additional_info.get('severity', 0) > 7:
            # Sort medicines to prioritize stronger ones
            # This is simplified - in a real system, you'd have potency data
            for medicine in refined_medicines:
                if 'ibuprofen' in medicine['name'].lower():
                    medicine['description'] = "PRIORITY: " + medicine['description']
        
        return refined_medicines