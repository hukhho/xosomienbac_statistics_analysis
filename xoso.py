import pandas as pd
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
import matplotlib.pyplot as plt
import math

# Constants
BET_AMOUNT = 1
WIN_RETURN = 97
BASE_NUM = 1

def kelly_criterion(win_probability, odds):
    bet_fraction = (win_probability * (odds + 1) - 1) / odds
    return bet_fraction


class OscarsGrindModel:
    def __init__(self, initial_bank_balance):
        self.current_date = "NULL"
        self.bank_balance = initial_bank_balance
        self.history = pd.DataFrame(columns=['date', 'bank_balance_before_betting', 'bank_balance_after_betting', 'betting_amount', 'result', 'current_cycle_profit'])
        self.betting_amount = BET_AMOUNT
        self.current_cycle_profit = 0
        self.total_bets = 0
        self.total_wins = 0
        self.loss_streak = 0
        self.win_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0

    def update_statistics(self, date, result, win_return, numbers_bet):
        self.current_date = date
        print(f"self.current_date: {self.current_date} date: {date}")
        total_betting_amount = len(numbers_bet) * self.betting_amount

        # Save the bank balance before betting
        bank_balance_before_betting = self.bank_balance

        # Update bank balance and current_cycle_profit
        if result:
            self.bank_balance += self.betting_amount * win_return - total_betting_amount
            self.current_cycle_profit += self.betting_amount * win_return - total_betting_amount
            print(f"{date} current_cycle_profit for win is {self.current_cycle_profit}")
        else:
            self.bank_balance -= total_betting_amount
            self.current_cycle_profit -= total_betting_amount
            print(f"{date}  current_cycle_profit for loss is {self.current_cycle_profit}")


        # Save the bank balance after betting
        bank_balance_after_betting = self.bank_balance

        # Update statistics
        self.total_bets += 1
        print(f"Total bets: {self.total_bets}")

        if result:
            self.total_wins += 1
            self.win_streak +=1
            self.loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.win_streak)

            print(f"Total wins: {self.total_wins}")
        else:
            self.win_streak = 0
            self.loss_streak += 1
            self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
            print(f"Loss streak: {self.loss_streak}, Max loss streak: {self.max_loss_streak}")

        self.history = self.history._append({'date': date, 
                                    'bank_balance_before_betting': bank_balance_before_betting,
                                    'bank_balance_after_betting': bank_balance_after_betting,
                                    'betting_amount': self.betting_amount,
                                    'result': result,
                                    'current_cycle_profit': self.current_cycle_profit}, 
                                ignore_index=True)
        
        
        
        #Update betting amount 
        ############################OSCARS GRIND 
        # if result:
        #     if self.current_cycle_profit == 1*BASE_NUM:
        #         self.betting_amount = 1*BASE_NUM
        #         self.current_cycle_profit = 0
        #     else:
        #         self.betting_amount = max(BASE_NUM, self.betting_amount - BASE_NUM)
        # else:
        #     self.betting_amount += BASE_NUM
        
        
        
        # -----------------------------------
        max_loss_streak_predict = 10
        stop_loss_streak = max_loss_streak_predict - 2
        cur_loss_streak = self.loss_streak
        if (cur_loss_streak <= stop_loss_streak):
            self.betting_amount = calculate_optimal_bet_size(current_loss_streak=cur_loss_streak, max_loss_streak=max_loss_streak_predict, bankroll=self.bank_balance, numbers_bet=51)
        else:
            self.betting_amount = 0
        # -----------------------------------
    
        #--------------------------------------------X2 bet if win   ()
        # max_win_streak_to_fight = 3
        
        # cur_win_streak = self.win_streak 
        
        # #1 Ex: win_streak = 1 => cur_win_streak = 1
        
        # if (0 < cur_win_streak <= max_win_streak_to_fight):
        #     self.betting_amount = math.floor(2*self.betting_amount)
        # else:
        #     self.betting_amount = 1
        #-----------------------------------------------
            
        # win_probability = 0.51
        # odds = 1

        # bet_fraction = kelly_criterion(win_probability, odds)
        # self.betting_amount = 1
            
        # if (testbetting > 0): 
        #     self.betting_amount = testbetting * 10
        # else:
        #     self.betting_amount = 0
            
        # print("Updated history DataFrame:")
        # print(self.history)

# The rest of the code remains unchanged.

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER
import calendar
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.platypus import SimpleDocTemplate, PageBreak

def calculate_optimal_bet_size_win(current_win_streak, max_win_streak, bankroll, numbers_bet):
    if current_win_streak >= max_win_streak:
        return 0

    # Calculate the remaining win streak allowed
    remaining_win_streak = max_win_streak - current_win_streak
    # Calculate the bet size, assuming you double the bet after each win
    bet_size = bankroll / (2**remaining_win_streak - 1) 
    # Calculate the bet size per number
    bet_size_per_number = bet_size / numbers_bet
    # Adjust the bet size per number if the total bet size exceeds the bankroll
    total_bet_size = bet_size_per_number * numbers_bet
    if total_bet_size > bankroll:
        bet_size_per_number = bankroll / numbers_bet
    # Ensure the bet size per number is divisible by 1
    bet_size_per_number = math.floor(bet_size_per_number / 1) * 1
    
    return bet_size_per_number

def calculate_optimal_bet_size(current_loss_streak, max_loss_streak, bankroll, numbers_bet):
    if current_loss_streak >= max_loss_streak:
        return 0

    # Calculate the remaining loss streak allowed
    remaining_loss_streak = max_loss_streak - current_loss_streak
    # Calculate the bet size, assuming you double the bet after each loss
    bet_size = bankroll / (2**remaining_loss_streak - 1) 
    # Calculate the bet size per number
    bet_size_per_number = bet_size / numbers_bet
    # Adjust the bet size per number if the total bet size exceeds the bankroll
    total_bet_size = bet_size_per_number * numbers_bet
    if total_bet_size > bankroll:
        bet_size_per_number = bankroll / numbers_bet
    # Ensure the bet size per number is divisible by 10
    bet_size_per_number = math.floor(bet_size_per_number / 1) * 1
    
    return bet_size_per_number

class OscarsGrindView:
    def generate_plots(self, model, plot_filename):
        plt.plot(model.history.index, model.history['bank_balance_after_betting'])
        plt.xlabel('Bets')
        plt.ylabel('Bank Balance')
        plt.title("Oscar's Grind Strategy")
        plt.savefig(plot_filename)
        plt.show()
        
    def generate_report(self, model, data, filename):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        title_style = styles["Heading1"]
        title_style.alignment = TA_CENTER
        report_title = Paragraph("Oscar's Grind Strategy Report", title_style)
        elements.append(report_title)
        elements.append(Spacer(1, 12))

        body_style = styles["BodyText"]
        elements.append(Paragraph(f"Total bets: {model.total_bets}", body_style))
        elements.append(Paragraph(f"Total wins: {model.total_wins}", body_style))
        elements.append(Paragraph(f"Win percentage: {model.total_wins / model.total_bets * 100:.2f}%", body_style))
        elements.append(Paragraph(f"Final bank balance: {model.bank_balance}", body_style))
        elements.append(Paragraph(f"Max loss streak: {model.max_loss_streak}", body_style))
        elements.append(Paragraph(f"Max win streak: {model.max_win_streak}", body_style))

        elements.append(Spacer(1, 12))
        summary_table = self.generate_table(model, data)
        
        # calendar = self.generate_calendar_table(model, data)

        # calendar_tables = self.generate_calendar_table(model, data)
        # # Append the tables to the story and add a PageBreak between each table
        # for idx, table in enumerate(calendar_tables):
        #     elements.append(table)
        #     if idx < len(calendar_tables) - 1:  # No need for a PageBreak after the last table
        #         elements.append(PageBreak())
        
        elements.append(Spacer(1, 12))
        elements.append(summary_table)

        doc.build(elements)
        
    def generate_table(self, model, data):
        table_data = [["Bet", "Date", "Numbers Bet", "Per", "Total", "Before", "After", "Result"]]
        table_data_colors = []
        
        for index, (_, row) in enumerate(data.iterrows()):
            numbers_bet = row["numbers_predict"]
            winning_number = row["winning_number"]

            if winning_number in numbers_bet:
                result = "Win"
                result_color = colors.green
            else:
                result = "Loss"
                result_color = colors.red
            table_data_colors.append(result_color)
            
            chunk_size = 10
            chunks = [numbers_bet[i:i + chunk_size] for i in range(0, len(numbers_bet), chunk_size)]

            final_text = ""
            for chunk in chunks:
                text = ", ".join(map(str, chunk))
                final_text += text + "\n"

            history_row = model.history.iloc[index]
            date_betting = history_row['date']
            bank_before_bet = history_row['bank_balance_before_betting']
            bet_per_number = history_row['betting_amount']
            total_bet = bet_per_number * len(numbers_bet)
            bank_after_bet = history_row['bank_balance_after_betting']

            table_data.append([index + 1, date_betting, final_text, bet_per_number, total_bet, bank_before_bet, bank_after_bet, result])

        table = Table(table_data, colWidths=[30, 70, 200, 50, 50, 50, 50, 50])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (2, 1), (2, -1), result_color),  # Set color for Numbers Bet column
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ('WORDWRAP', (2, 1), (2, -1))  # Allowing word wrap for the "Numbers Bet" column
        ]))
        
           # Apply text colors for the result column
        for i, color in enumerate(table_data_colors, start=1):
            table.setStyle(TableStyle([
                ('BACKGROUND', (-1, i), (-1, i), color),
        ]))

        return table

    def generate_calendar_table(self, model, data):
        color_dict = {'Win': colors.green, 'Loss': colors.red}

        # Group data by year and month
        data_by_month = {}
        for index, (_, row) in enumerate(data.iterrows()):
            history_row = model.history.iloc[index]
            date = history_row['date']
            year, month, _day = date.split('-')
            year_month = f"{year}-{month}"
            data_by_month.setdefault(year_month, []).append((history_row, row))

        calendar_tables = []
        for year_month, month_data in data_by_month.items():
            year, month = map(int, year_month.split('-'))

            # Create a calendar object for the given month and year
            cal = calendar.Calendar()
            month_days = cal.monthdayscalendar(year, month)

            # Prepare table data
            table_data = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
            for week in month_days:
                week_data = []
                for day in week:
                    if day == 0:
                        week_data.append("")  # Add empty cell for days not in the month
                    else:
                        date = f"{year}-{month:02d}-{day:02d}"
                        result = None

                        # Find the corresponding date in the dataset
                        for history_row, row in month_data:
                            if history_row['date'] == date:
                                result = row['winning_number'] in row['numbers_predict']
                                break

                        week_data.append(result)

                table_data.append(week_data)

            # Create the table
            table = Table(table_data)

            # Apply table styles
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 12),
                ('TOPPADDING', (0, 1), (-1, -1), 12),
            ]))

            # Apply text colors and results for the result column
            for row, week in enumerate(table_data[1:], start=1):
                for col, result in enumerate(week, start=0):
                    if result is not None:
                        print(f"for col, result in enum result: {result}")
                        text = "Win" if result else "Loss"
                        color = color_dict['Win'] if result else color_dict['Loss']
                        table.setStyle(TableStyle([
                            ('TEXTCOLOR', (col, row), (col, row), color),
                            ('TEXT', (col, row), (col, row), text),
                        ]))

            calendar_tables.append(table)

        return calendar_tables
            
    def generate_winloss_table(self, model, data):
        # Create a color dictionary for result colors
        table_data = [["Date", "Result"]]
        table_data_colors = []
        color_dict = {'Win': colors.green, 'Loss': colors.red}
        
        for index, (_, row) in enumerate(data.iterrows()):
            numbers_bet = row["numbers_predict"]
            winning_number = row["winning_number"]

            if winning_number in numbers_bet:
                result = "Win"
                color = color_dict['Win']
            else:
                result = "Loss"
                color = color_dict['Loss']

            history_row = model.history.iloc[index]
            date = history_row['date'] 
               
            table_data_colors.append(color)
            table_data.append([date,result])
            
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 12),
        ]))

        # Apply text colors for the result column
        for i, color in enumerate(table_data_colors, start=1):
            table.setStyle(TableStyle([
                ('TEXTCOLOR', (1, i), (1, i), color),
            ]))

        return table  
    
# OscarsGrindController
class OscarsGrindController:
    def __init__(self, model, view, data):
        self.model = model
        self.view = view
        self.data = data

    def run_simulation(self):
        for _, row in self.data.iterrows():
            numbers_bet = row["numbers_predict"]
            winning_number = row["winning_number"]
            date = row["date_column"]
            result = 1 if winning_number in numbers_bet else 0
            
            # self.model.update_bank(result, WIN_RETURN, numbers_bet)
            # self.model.update_betting_amount(result)
            #self.model.update_statistics(date, result, self.model.betting_amount)
            self.model.update_statistics(date, result, WIN_RETURN, numbers_bet)

    def generate_report(self, filename):
        self.view.generate_report(self.model, self.data, filename)

    def generate_plots(self, filename):
        self.view.generate_plots(self.model, filename)



def read_csv_file(file_name):
    df = pd.read_csv(file_name, converters={'numbers_predict': lambda x: list(map(int, x.strip('[]').split(',')))})
    return df

import random

from datetime import datetime, timedelta

# Main
def main():
    input_file = "combined2003to2023.csv"
    input_data = read_csv_file(input_file)
    # input_data1 = pd.read_csv(input_file)
        
    # Initialize model, view, and controller
    model = OscarsGrindModel(initial_bank_balance=100_000)
    view = OscarsGrindView()
    controller = OscarsGrindController(model, view, input_data)

    # Run simulation
    controller.run_simulation()

    # Generate report and plots
    
    timestamp = time.time()

    controller.generate_report(f"OscarsGrindReport_{input_file}_{timestamp}.pdf")
    controller.generate_plots(f"OscarsGrindPlot_{input_file}_{timestamp}.png")
    
if __name__ == "__main__":
    main()