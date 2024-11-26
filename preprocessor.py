import pandas as pd
import re
def write_df_to_file(data_frame, filename='output.csv', **kwargs):
    try:
        data_frame.to_csv(filename, **kwargs)
        print(f"DataFrame successfully written to {filename}")
    except Exception as e:
        print(f"Error writing DataFrame to file: {e}")


def preprocess(data):
    lines = data.split('\n')
    print("lines", lines)
    chat_data = []
    for line in lines:
        if re.match(r'\d+/\d+/\d+, \d+:\d+\s*([APap][Mm])? - .+: .+', line):
            date, user_message = re.split(' - ', line, 1)
            date = date.strip()
            user, message = user_message.split(': ', 1)
            chat_data.append([date, user, message])
    

    df = pd.DataFrame(chat_data, columns=['date', 'user', 'message'])
    df['datetime'] = pd.to_datetime(df['date'], format='mixed')  # Updated date and time format
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.strftime('%B')
    df['day_name'] = df['datetime'].dt.strftime('%A')
    df['period'] = df['datetime'].dt.strftime('%p')
    df['only_date'] = df['datetime'].dt.date
    return df
