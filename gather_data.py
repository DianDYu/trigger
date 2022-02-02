import csv


def process_data(filename, out_file_name):
    csv_file = open(out_file_name, mode="w")
    fieldnames = ['human_utterance', 'bot_utterance', 'label', 'human_context', 'bot_context', 'persona']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    with open(filename) as f:
        turn = 0  # start with 0, each turn include two utterances (one user, one bot)
        prev_persona = ""
        context = []
        for line in f:
            line_info = line.split("\t")  # text, labels, episode_done, speaker, persona
            text = line_info[0][5:]
            label = line_info[1][7:]  # __ok__, __notok__
            speaker = line_info[3].split(":")[1]  # human, bot
            persona_info = line_info[4].split("your persona:")  # in the format of "bot_persona:your persona:" or "bot_persona:nan"

            current_utterance = text.split("\\n")[-1]  # this is kind of weird in the original data if read in python
            if len(persona_info) == 1:
                persona_1 = "nan"
                persona_2 = "nan"
            else:
                persona_1 = persona_info[1].strip()
                if persona_1.endswith("\\n"):
                    persona_1 = persona_1[:-2]  # also weird
                persona_2 = persona_info[2].strip()

            if persona_info != prev_persona:  # new conversation
                turn = 0
                context = []
                prev_persona = persona_info
                assert speaker == "human", "human should be the first speaker"

            if speaker == "bot":
                # ignore if a classifier is applied in the data for the human input
                if current_utterance.startswith("Hey do you want to talk about something else?"):
                    continue
                new_row = dict()
                new_row['human_utterance'] = context[-1]
                new_row['bot_utterance'] = current_utterance
                if len(context) >= 4:
                    bot_context = "\n".join(i for i in context[-4:])  # last four turns as used in the safety paper
                    human_context = "\n".join(i for i in context[-4:-1])
                else:
                    bot_context = "\n".join(i for i in context)
                    human_context = "\n".join(i for i in context[:-1])
                if len(human_context.strip()) == 0:
                    human_context = "None\n"
                new_row['human_context'] = human_context
                new_row['bot_context'] = bot_context
                new_row['persona'] = persona_1 + "\n" + persona_2
                new_row['label'] = label

                writer.writerow(new_row)
                turn += 1

            context.append(current_utterance)


def main():
    for data_path in [train_path, dev_path, test_path]:
        data_ext = data_path.split("/")[-1].split(".txt")[0]
        out_data_path = "./data/%s.csv" % data_ext
        process_data(data_path, out_data_path)


if __name__ == "__main__":
    main()
