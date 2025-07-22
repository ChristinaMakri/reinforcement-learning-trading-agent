import numpy as np

class TradingEnv:
    def __init__(self, data, initial_balance=1_000_000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.total_profit = 0
        self.avg_buy_price = 0  # Μέση τιμή αγοράς θέσεων
        return self._get_state()

    def _get_state(self):
        price = self.data['Close'].iloc[self.current_step] / 100_000  # Κανονικοποίηση τιμής
        balance = self.balance / 1_000_000  # Κανονικοποίηση υπολοίπου
        return np.array([price, balance, self.position])

    def step(self, action):
        """
        Actions: 0 = Buy, 1 = Sell, 2 = Hold
        """
        price = self.data['Close'].iloc[self.current_step]
        reward = 0

        if action == 0:  # Buy
            if self.balance >= price:
                # Ενημέρωση μέσης τιμής αγοράς
                total_cost = self.avg_buy_price * self.position + price
                self.position += 1
                self.balance -= price
                self.avg_buy_price = total_cost / self.position

        elif action == 1:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += price
                # Υπολογισμός κέρδους από την πώληση
                profit = price - self.avg_buy_price
                self.total_profit += profit
                reward = profit
                # Αν πουλήθηκε όλη η θέση, μηδενίζουμε μέση τιμή αγοράς
                if self.position == 0:
                    self.avg_buy_price = 0

        else:  # Hold
            if self.position > 0:
                unrealized_profit = (price - self.avg_buy_price) * self.position
                # Μικρή ανταμοιβή για κερδοφόρες ανοιχτές θέσεις
                reward = 0.01 * unrealized_profit

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self._get_state()

        return next_state, reward, done
