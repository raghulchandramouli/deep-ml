import numpy as np

def derivative(coeffs):
    """
    A place for calc derivatives quickly
    """
    result = []
    for i in range(1, len(coeffs)):
        result.append(i * coeffs[i])
    
    if len(result) == 0:
        return [0.0]
        
    return result

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """

    f_prime = derivative(f_coeffs)
    g_prime = derivative(g_coeffs)

    term1 = np.convolve(f_prime, g_coeffs)
    term2 = np.convolve(f_coeffs, g_prime)

    # we add or simply put concatenate these result
    result = term1 + term2

    return [round(x, 4) for x in result.tolist()]