-- SQL examples used for analysis/interview practice

-- 1. Average premium by smoker status
SELECT smoker, AVG(charges) AS avg_premium
FROM insurance
GROUP BY smoker;

-- 2. Average premium by region
SELECT region, AVG(charges) AS avg_premium
FROM insurance
GROUP BY region
ORDER BY avg_premium DESC;

-- 3. High-premium customers
SELECT *
FROM insurance
WHERE charges > 20000
ORDER BY charges DESC;

-- 4. Average BMI and premium by sex
SELECT sex, AVG(bmi) AS avg_bmi, AVG(charges) AS avg_premium
FROM insurance
GROUP BY sex;
