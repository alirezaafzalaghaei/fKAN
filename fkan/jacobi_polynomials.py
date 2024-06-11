def jacobi_polynomial(x, n, alpha, beta, gamma, a, b):
    if n == 0:
        return x / (x + 1e-7)
    elif n == 1:
        return (
            alpha - beta + (alpha + beta + 2) * (2 * x**gamma - a - b) / (b - a)
        ) / 2
    elif n == 2:
        return (
            ((alpha + 1) * (alpha + 2)) / 2
            + (
                (alpha + 2)
                * (3 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1)
            )
            / 2
            + (
                (3 + alpha + beta)
                * (4 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 2
            )
            / 8
        )
    elif n == 3:
        return (
            ((alpha + 1) * (alpha + 2) * (3 + alpha)) / 6
            + (
                (alpha + 2)
                * (3 + alpha)
                * (4 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1)
            )
            / 4
            + (
                (3 + alpha)
                * (4 + alpha + beta)
                * (5 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 2
            )
            / 8
            + (
                (4 + alpha + beta)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 3
            )
            / 48
        )
    elif n == 4:
        return (
            ((alpha + 1) * (alpha + 2) * (3 + alpha) * (4 + alpha)) / 24
            + (
                (alpha + 2)
                * (3 + alpha)
                * (4 + alpha)
                * (5 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1)
            )
            / 12
            + (
                (3 + alpha)
                * (4 + alpha)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 2
            )
            / 16
            + (
                (4 + alpha)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 3
            )
            / 48
            + (
                (5 + alpha + beta)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 4
            )
            / 384
        )
    elif n == 5:
        return (
            ((alpha + 1) * (alpha + 2) * (alpha + 3) * (alpha + 4) * (alpha + 5)) / 120
            + (
                (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1)
            )
            / 48
            + (
                (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 2
            )
            / 48
            + (
                (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 3
            )
            / 96
            + (
                (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 4
            )
            / 384
            + (
                (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 5
            )
            / 3840
        )
    elif n == 6:
        return (
            (
                (alpha + 1)
                * (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
            )
            / 720
            + (
                (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1)
            )
            / 240
            + (
                (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 2
            )
            / 192
            + (
                (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 3
            )
            / 288
            + (
                (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 4
            )
            / 768
            + (
                (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * (11 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 5
            )
            / 3840
            + (
                (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * (11 + alpha + beta)
                * (12 + alpha + beta)
                * ((2 * x**gamma - a - b) / (b - a) - 1) ** 6
            )
            / 46080
        )
    elif n > 6:
        raise ValueError(
            f"The current implementation supports a maximum degree of 6, but you entered {n}. Higher degrees may lead to numerical instabilities, overfitting, and increased computational complexity. Please consider using a lower degree."
        )
    elif n < 0:
        raise ValueError(
            "Degrees must be non-negative. Negative degrees are not allowed."
        )
