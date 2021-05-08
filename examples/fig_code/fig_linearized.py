import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")

def main():
	# Problem parameters.
	alpha = 0.25
	beta = 0.5
	gamma = 0
	h_prev = 20

	# Plot parameters.
	y_eps = 5
	y_min = -30 - y_eps
	y_max = 30 + y_eps

	x_eps = 0.5
	x_min = 0 - x_eps
	x_max = 10 + x_eps

	# Health dynamics equation.
	d = np.linspace(0, 10)
	f = h_prev - alpha*d - beta*d**2 + gamma

	fig1 = plt.figure()
	fig1.set_size_inches(10,8)
	plt.plot(d, f, "b-", label = "Original ($f_{ti}$)")
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xlabel("$d_{ti}$")
	plt.legend()
	plt.show()
	
	# Linearization.
	d_lin_idx = int(0.35*len(d))
	d_lin = d[d_lin_idx]
	f_hat = h_prev - alpha*d - beta*d_lin*(2*d - d_lin) + gamma
	
	fig2 = plt.figure()
	fig2.set_size_inches(10,8)
	plt.plot(d, f, "b-", label = "Original ($f_{ti}$)")
	plt.plot(d, f_hat, "r-", label = "Linearized ($\hat f_{ti}$)")

	# Label linearization point.
	plt.plot(d_lin, f[d_lin_idx], "bo", markersize = 6)
	plt.vlines(d_lin, y_min, f[d_lin_idx], lw = 1, ls = ':', color = "grey")
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xlabel("$d_{ti}$")
	
	ax = plt.gca()
	x_locs = list(ax.get_xticks())
	x_locs = x_locs[1:-1]
	ax.set_xticks(x_locs + [d_lin])
	ax.set_xticklabels([int(x) for x in x_locs] + ["$d_{ti}^s$"])
	plt.legend()
	plt.show()

	# Linearization with slack.
	slack = 5
	f_hat_slack = f_hat - slack

	fig3 = plt.figure()
	fig3.set_size_inches(10,8)
	plt.plot(d, f, "b-", label = "Original ($f_{ti}$)")
	plt.plot(d, f_hat, "r-", label = "Linearized ($\hat f_{ti}$)")
	plt.plot(d, f_hat_slack, "r--", label = "Linearized with Slack ($\hat f_{ti} - \delta_{ti}$)")

	# Label linearization point.
	plt.plot(d_lin, f[d_lin_idx], "bo", markersize = 6)
	plt.vlines(d_lin, y_min, f[d_lin_idx], lw = 1, ls = ':', color = "grey")
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xlabel("$d_{ti}$")

	ax = plt.gca()
	x_locs = list(ax.get_xticks())
	x_locs = x_locs[1:-1]
	ax.set_xticks(x_locs + [d_lin])
	ax.set_xticklabels([int(x) for x in x_locs] + ["$d_{ti}^s$"])
	plt.legend()

	# Label slack gap.
	d_shift_idx = int(0.8*len(d))
	d_shift = d[d_shift_idx]
	plt.vlines(d_shift, f_hat_slack[d_shift_idx], f_hat[d_shift_idx], ls = "-.")
	plt.text(d_shift + 0.25, f_hat_slack[d_shift_idx] + slack/4, "$\delta_{ti}$", size = "large")
	# plt.vlines(d_shift, f_hat_slack[d_shift_idx], f_hat[d_shift_idx], ls = "-.", color = "limegreen")
	# plt.text(d_shift + 0.25, f_hat_slack[d_shift_idx] + slack/3, "$\delta_{ti}$", color = "limegreen", size = "large")
	plt.show()

if __name__ == '__main__':
	main()
