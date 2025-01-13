import requests
import json
import time
import subprocess

# URL of the service
url = "https://rag.ki.rtr.at/llm-service/chat"

# questions
faq = ['Ich möchte meinen Mitarbeitern den Einsatz von ChatGPT ermöglichen. Welche Rolle im AI Act nehme ich als Unternehmen ein?',
'Ich möchte Werbeaussendungen mit einem Large Language Model automatisch generieren und individuell angepasst an Kunden versenden. Muss ich das offenlegen?',
'Ich möchte die Arbeitsleistung meiner Mitarbeiter im Home Office mittels eines KI-Systems auswerten. Wie ist dieses System einzustufen?',
'Ich möchte KI-Systeme in meinem Unternehmen einsetzen. Gibt es dafür bestimmte Auflagen, die ich erfüllen muss?',
'Ich möchte auf meiner Website eine  KI-generierte Hintergrundmusik implementieren?  Unterliege ich etwaigen Verpflichtungen nach dem AI Act?',
'Ich möchte gerne KI-Reallabore nutzen. Ab wann ist das möglich und wie genau funktionieren diese?',
'Ich möchte meine Jobinserate mittels KI generieren lassen. Ist das zulässig? Was muss ich beachten?',
'Ich möchte KI beim Personal-Recruiting einsetzen und spezielle Recruitingsoftware dafür anschaffen. Was muss ich beachten?',
'Ich habe mein eigenes KI-System/KI-Modell für den internen Gebrauch im Unternehmen entwickelt. Gilt der AI-Act auch für mich?',
'Ich möchte KI-generierte Videos von einem Anbieter beziehen und auf meiner eigenen Plattform veröffentlichen, was muss ich aus Sicht des AI Acts beachten?',
'Ich möchte in meinem Unternehmen KI einsetzen. Muss ich eine KI-Policy erlassen?',
'Ich möchte im Bewerbungsprozess ein KI-System einsetzen, das eine Risikobewertung durchführt. Das KI-System soll insbesondere bewerten ob der/die Bewerber/in eine Neigung zur Begehung von Straftaten hat. Was muss ich da beachten?',
'Ich möchte ein KI-System einsetzen, dass die Emotionen meiner Mitarbeiter während des Arbeitens am PC/Laptop abliest und bewertet. Was muss ich da beachten?',
'Ich möchte in meinem Unternehmen ein KI-System im Bereich der kritischen Infrastruktur verwenden. Ab wann handelt es sich um ein Hochrisiko-KI-System? ',
'Ich bin ein Versicherungsmakler und berechne mit Hilfe eines gekauften KI-Systems die Kreditwürdigkeit meiner Kunden. Welche Rolle nach dem AI-Act nehme ich als Unternehmer ein? Treffen mich besondere Verpflichtungen? Welcher Risikostufe sind derartige KI-Systeme zuzuordnen?',
'Ich möchte die biometrischen Daten meiner Kunden mit Hilfe eines KI-Systems sammeln um anhand dieser Daten Rückschlüsse auf ihre Rasse, Religionszugehörigkeit und politische Einstellung ziehen zu können. Diese Daten möchte ich für zielgerichtete Werbung benutzen. Handelt es sich dabei um eine nach dem AI-Act erlaubte KI-Nutzung?',
'Eine FH möchte ein KI-System verwenden, welche die Zulassung ausländischer Studenten prüft. Kann man hier ein KI-System ohne Weiteres einsetzen oder ist eine vorherige Genehmigung notwendig? Welcher Risikostufe unterliegt so ein KI-System?',
'Eine FH möchte mit Hilfe eines KI-Systems die Lernerfolge der Studenten aufzeichnen um am Ende des Studienjahres den Studenten zu einem passenden Praktikumsplatz für das kommende Studienjahr zuzuordnen. Ist ein solches KI-System genehmigungspflichtig? Welcher Risikostufe unterliegt ein solches System?',
'Ich bin ein Kleinunternehmer und möchte beim Bewerbungsgespräch KI-gesteuerte Lügendetektoren einsetzen. Ist das erlaubt? ',
'Mein Arbeitgeber wertet den Gemütszustand seiner Mitarbeiter am Arbeitsplatz mittels eines eigens dafür angeschafften KI-Systems aus. Ist das erlaubt? ',
'Ich habe ein KI-Modell entwickelt, dass Patienteninformationen sammelt und in eine Datenbank speichert. Dieses KI-System möchte ich Arztpraxen zur Verfügung stellen. Was muss ich beachten?',
'Ich bin ein Kleinunternehmer und hätte gerne Zugang zu den Reallaboren. Was muss ich tun um Zugang zu diesen zu bekommen bzw bin ich überhaupt berechtigt Zugang zu einem Reallabor zu erhalten?',
'Ich bin eine Kleinunternehmerin und verkaufe in meinem Onlineshop ausschließlich Frauenbekleidung. Je nach Verkaufsverhalten meiner Kundinnen liefert, das sich im Einsatz befindliche KI-Modell personalisierte Bekleidungsvorschläge, die den Kundinnen direkt auf der Webseite angezeigt werden. Bin ich als Betreiberin gemäß dem AI-Act einzustufen? Was muss ich beachten?',
]

# Container for the final JSON data
output_data = []

for q in faq:
    # Data payload
    payload = {
        "prompt": q
    }

    # Headers, based on your request
    headers = {
        "Accept": "text/event-stream",
        "Authorization": "Basic cnRyLWFpOnJ0ci1haQ==",
        "Content-Type": "application/json"
    }

    # Send the POST request
    s = requests.Session()
    response = s.post(url, json=payload, headers=headers, stream=True)

    # Check for a successful response
    if response.status_code == 200:
        events = []
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    # Strip 'data: ' part from the line
                    json_data = decoded_line[6:]
                    event_data = json.loads(json_data)
                    events.append(event_data)
                    
                    # Check to see if we have enough events
                    if len(events) >= 3:
                        break

            # Introduce a delay, as subsequent responses may take time
            time.sleep(1)

        # Process the second event
        main_event = events[2]
        content = json.loads(main_event["content"])

        # Prepare data for JSON
        question_data = {"question": q, "main_titles": []}

        for item in content:
            main_title = item.get('title', '<No Title>')
            main_data = {"title": main_title, "relevant_chunks": []}
            for chunk in item.get('relevantChunks', []):
                chunk_title = chunk.get('title', '<No Title>')
                main_data["relevant_chunks"].append(chunk_title)
            question_data["main_titles"].append(main_data)

        output_data.append(question_data)
        
    else:
        print('Request failed with status code:', response.status_code)

# Write the collected data to a JSON file
with open("//sshare1.rtr.at/p-lw/bma/System/Documents/GitHub/aia-rag/auto_test/faq_output.json", "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)

print("JSON file 'faq_output.json' created successfully.")

def git_push():
    try:
        # Add the updated file to the staging area
        subprocess.run(["git", "add", "faq_output.json"], check=True)

        # Commit the changes with a message
        subprocess.run(["git", "commit", "-m", "Update FAQ JSON file"], check=True)

        # Push the changes to the remote repository
        subprocess.run(["git", "push"], check=True)

        print("File successfully pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Git operations: {e}")

# Call the function to push changes
git_push()
