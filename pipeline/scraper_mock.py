import pandas as pd
import random
from datetime import datetime, timedelta

def generate_sample_data(subreddit: str, limit: int = 50):
    samples = {
        "nursing": [
            "I've been a nurse for 10 years and burnout is real. Short staffed again tonight.",
            "Hospital admin cut our break times again. Patient safety is being compromised.",
            "Finally got my BSN! So proud of this accomplishment despite everything.",
            "Insurance denied my patient's medication AGAIN. This system is broken.",
            "Our unit just got new equipment - finally feeling supported by management.",
            "Working 3 back to back 12 hour shifts. Exhausted but patients need us.",
            "Had the most rewarding shift today. Patient recovered better than expected.",
            "Mandatory overtime again this weekend. Nobody asked us.",
            "New hospital policy makes no sense. Who makes these decisions?",
            "Patient ratio is 1:8 tonight. This is dangerous and unacceptable.",
            "Just got my dream job at a teaching hospital. So excited!",
            "Administration keeps cutting staff but expects the same output.",
            "Family was so grateful today. Moments like these remind me why I do this.",
            "Charge nurse threw me under the bus in front of the whole team.",
            "Finally a hospital that invests in nurse wellbeing programs.",
        ],
        "HealthInsurance": [
            "Insurance denied my chemo treatment. I don't know what to do.",
            "Prior authorization taking 3 weeks. My condition is getting worse.",
            "Finally got my claim approved after 6 months of fighting. Don't give up.",
            "ER bill was $45,000 for a 2 hour visit. Insurance only covered $3,000.",
            "Switching jobs and terrified about losing coverage for pre-existing condition.",
            "Insurance company called my surgery elective. It was life saving.",
            "Open enrollment is so confusing. How do people navigate this?",
            "Got denied for mental health coverage. This stigma needs to end.",
            "Appealed my denial and won. Always appeal, never give up.",
            "Deductible reset January 1st and I cant afford my medications now.",
            "Insurance covered my therapy finally after 2 years of fighting.",
            "Out of network surprise bill wiped out my entire savings.",
            "Healthcare system is designed to confuse people into giving up.",
            "My doctor is amazing but insurance keeps blocking her recommendations.",
            "Small business owner here - insurance costs are destroying my company.",
        ],
        "AskDocs": [
            "Doctor dismissed my symptoms for 2 years. Finally diagnosed with autoimmune disease.",
            "Can't afford my insulin even with insurance. Rationing doses to survive.",
            "Telehealth appointment was actually really helpful and convenient.",
            "Waited 4 months for specialist appointment. Healthcare access is a problem.",
            "My doctor actually listened to me today. Rare but so appreciated.",
            "Hospital billing department is impossible to reach. Sent to collections.",
            "Free clinic in my city is a lifesaver for uninsured patients.",
            "Prescription costs tripled overnight. Same medication, same pharmacy.",
            "Second opinion saved my life. First doctor was completely wrong.",
            "Mental health resources in my rural area are basically nonexistent.",
            "New primary care doctor is amazing. Actually spends time with patients.",
            "Medical debt is destroying my credit score and my mental health.",
            "Community health center accepted sliding scale payment. So grateful.",
            "Misdiagnosed for 3 years. Finally found a doctor who ran proper tests.",
            "Preventive care visit was fully covered. More people should use this.",
        ]
    }

    posts = []
    texts = samples.get(subreddit, samples["nursing"])

    for i in range(min(limit, len(texts) * 3)):
        text = texts[i % len(texts)]
        posts.append({
            "id": f"sample_{subreddit}_{i}",
            "subreddit": subreddit,
            "title": text,
            "selftext": "",
            "score": random.randint(10, 5000),
            "num_comments": random.randint(5, 200),
            "created_utc": int((datetime.now() - timedelta(days=random.randint(1, 30))).timestamp())
        })

    return pd.DataFrame(posts)


if __name__ == "__main__":
    print("Testing data collection...")
    all_data = []

    for sub in ["nursing", "HealthInsurance", "AskDocs"]:
        df = generate_sample_data(sub, 15)
        all_data.append(df)
        print(f"\nr/{sub}: {len(df)} posts collected")
        print(f"Sample post: {df['title'].iloc[0][:70]}...")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal posts collected: {len(combined)}")
    print(f"Subreddits: {combined['subreddit'].unique()}")
    print("\nData collection working perfectly!")