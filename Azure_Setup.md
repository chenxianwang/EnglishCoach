# Getting your Azure Speech key (free)

The 5-second limit you saw is only the Speech Studio web demo. The API's free
tier (**F0**) gives **5 audio hours/month** including pronunciation assessment —
no per-clip limit. Here's how to get a key.

## 1. Create a free Azure account
- Go to https://azure.microsoft.com/free and sign up.
- A card is requested for identity only; the **F0** Speech tier is free and
  won't auto-charge. (You can also set a $0 spending cap.)

## 2. Create a Speech resource
- In the Azure Portal (https://portal.azure.com), search **"Speech services"**
  → **Create**.
- Fill in:
  - **Subscription:** your free one
  - **Resource group:** create one, e.g. `english-coach`
  - **Region:** pick one near you, e.g. `eastus` or `southeastasia`
    (remember this — it's your `AZURE_SPEECH_REGION`)
  - **Name:** anything, e.g. `english-coach-speech`
  - **Pricing tier:** select **Free F0**
- Click **Review + create** → **Create**.

## 3. Copy your key and region
- Open the resource → left menu **"Keys and Endpoint"**.
- Copy **KEY 1** (that's `AZURE_SPEECH_KEY`) and note the **Location/Region**
  (that's `AZURE_SPEECH_REGION`, e.g. `eastus`).

## 4. Set them in your terminal
```bash
export AZURE_SPEECH_KEY=your_key_here
export AZURE_SPEECH_REGION=eastus
pip install azure-cognitiveservices-speech faster-whisper anthropic
```

## 5. Run the real pronunciation assessment
Read a script aloud, save the script as `script.txt`, then:
```bash
python english_coach.py recording.m4a --reference script.txt --azure --out report.html
```
The report's "Pronunciation score (Azure)" section will show your real
accuracy / fluency / completeness / prosody scores and per-word error flags —
exactly like the Speech Studio demo, but on your full recording.

## Notes
- `--azure` requires `--reference` (Azure scores you against the script you read).
- Without `--azure`, the app still runs and uses the free transcript-diff
  pronunciation estimate.
- Cost: F0 = free up to 5 hours/month. A 2-minute clip uses ~0.03 hours, so you
  can run it ~150 times a month at no cost.
