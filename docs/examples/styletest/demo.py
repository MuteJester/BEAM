from _docs import figure, save_data  # hide

squares = [n * n for n in range(1, 6)]
print("squares:", squares)
print(f"sum = {sum(squares)}")

# --- hide
with figure("squares") as (fig, ax):
    ax.plot(range(1, 6), squares, marker="o")
    ax.set_xlabel("n")
    ax.set_ylabel("n squared")
save_data("squares", {"n": list(range(1, 6)), "square": squares})
# --- show
