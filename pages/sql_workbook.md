# SQL COMPLETE WORKBOOK: Basic to Advanced

## 1. SELECT & WHERE - Getting Data
**Intuition**: Ask for specific columns and filter rows that meet conditions.

```sql
-- Basic SELECT
SELECT name, age FROM users;
SELECT * FROM users WHERE age > 25;
SELECT DISTINCT department FROM employees;

-- Multiple conditions
SELECT * FROM orders WHERE status = 'shipped' AND total > 100;
SELECT * FROM products WHERE category IN ('electronics', 'books');
SELECT * FROM users WHERE name LIKE 'J%';  -- starts with J
SELECT * FROM items WHERE price BETWEEN 10 AND 50;
SELECT * FROM records WHERE value IS NULL;
```

## 2. ORDER BY & LIMIT - Sorting & Limiting
**Intuition**: Control how results are ordered and how many rows you get.

```sql
SELECT * FROM products ORDER BY price DESC LIMIT 10;
SELECT * FROM users ORDER BY last_name ASC, first_name ASC;
SELECT * FROM sales ORDER BY date DESC OFFSET 20 LIMIT 10;  -- pagination
```

## 3. AGGREGATE FUNCTIONS - Summarizing Data
**Intuition**: Collapse many rows into summary statistics.

```sql
SELECT COUNT(*) FROM orders;
SELECT AVG(salary), MIN(salary), MAX(salary), SUM(salary) FROM employees;
SELECT COUNT(DISTINCT customer_id) FROM orders;
```

## 4. GROUP BY & HAVING - Grouping & Filtering Groups
**Intuition**: GROUP BY creates buckets, aggregates summarize each bucket, HAVING filters buckets.
**Rule**: Columns in SELECT must be in GROUP BY or inside aggregate functions.

```sql
-- Count orders per customer
SELECT customer_id, COUNT(*) as order_count 
FROM orders 
GROUP BY customer_id;

-- Average salary by department, only departments with avg > 60k
SELECT department, AVG(salary) as avg_sal
FROM employees
GROUP BY department
HAVING AVG(salary) > 60000;

-- Multiple grouping columns
SELECT department, job_title, COUNT(*) as count
FROM employees
GROUP BY department, job_title;
```

## 5. JOINS - Combining Tables
**Intuition**: Match rows from different tables based on related columns.

```sql
-- INNER JOIN: Only matching rows from both tables
SELECT u.name, o.order_date, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN: All rows from left table, nulls if no match on right
SELECT u.name, o.total
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN: All rows from right table
SELECT u.name, o.total
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;

-- FULL OUTER JOIN: All rows from both tables
SELECT u.name, o.total
FROM users u
FULL OUTER JOIN orders o ON u.id = o.user_id;

-- CROSS JOIN: Cartesian product (every combination)
SELECT a.color, b.size FROM colors a CROSS JOIN sizes b;

-- Self JOIN: Join table to itself
SELECT e.name as employee, m.name as manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- Multiple JOINs
SELECT u.name, o.order_date, p.product_name, oi.quantity
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id;
```

## 6. SUBQUERIES - Queries Inside Queries
**Intuition**: Use query results as input to another query.

```sql
-- Subquery in WHERE
SELECT name FROM users 
WHERE id IN (SELECT user_id FROM orders WHERE total > 1000);

-- Subquery in SELECT (must return single value)
SELECT name, (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count
FROM users u;

-- Subquery in FROM (derived table)
SELECT avg_salary_by_dept.department, avg_salary_by_dept.avg_sal
FROM (
    SELECT department, AVG(salary) as avg_sal
    FROM employees
    GROUP BY department
) as avg_salary_by_dept
WHERE avg_salary_by_dept.avg_sal > 70000;

-- Correlated subquery (inner query references outer query)
SELECT name, salary
FROM employees e1
WHERE salary > (SELECT AVG(salary) FROM employees e2 WHERE e2.department = e1.department);
```

## 7. COMMON TABLE EXPRESSIONS (CTEs) - Named Subqueries
**Intuition**: Create named temporary result sets for cleaner, readable queries.

```sql
-- Single CTE
WITH high_value_customers AS (
    SELECT user_id, SUM(total) as lifetime_value
    FROM orders
    GROUP BY user_id
    HAVING SUM(total) > 5000
)
SELECT u.name, hvc.lifetime_value
FROM users u
JOIN high_value_customers hvc ON u.id = hvc.user_id;

-- Multiple CTEs
WITH 
sales_by_month AS (
    SELECT DATE_TRUNC('month', order_date) as month, SUM(total) as monthly_sales
    FROM orders
    GROUP BY month
),
avg_monthly_sales AS (
    SELECT AVG(monthly_sales) as avg_sales FROM sales_by_month
)
SELECT sbm.month, sbm.monthly_sales, ams.avg_sales
FROM sales_by_month sbm
CROSS JOIN avg_monthly_sales ams
WHERE sbm.monthly_sales > ams.avg_sales;
```

## 8. WINDOW FUNCTIONS - Analytics Without Grouping
**Intuition**: Perform calculations across rows while keeping individual rows (unlike GROUP BY).
**Rule**: OVER() clause defines the "window" of rows to calculate over.

```sql
-- ROW_NUMBER: Assign sequential numbers
SELECT name, salary, 
       ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- RANK & DENSE_RANK: Handle ties differently
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank,          -- 1,2,2,4
       DENSE_RANK() OVER (ORDER BY salary DESC) as d_rank   -- 1,2,2,3
FROM employees;

-- PARTITION BY: Separate windows per group
SELECT name, department, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
FROM employees;

-- Running totals with window frames
SELECT order_date, total,
       SUM(total) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total
FROM orders;

-- LAG & LEAD: Access previous/next row values
SELECT order_date, total,
       LAG(total, 1) OVER (ORDER BY order_date) as prev_total,
       LEAD(total, 1) OVER (ORDER BY order_date) as next_total
FROM orders;

-- NTILE: Divide into buckets
SELECT name, salary,
       NTILE(4) OVER (ORDER BY salary) as quartile
FROM employees;

-- Moving averages
SELECT order_date, total,
       AVG(total) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7day
FROM orders;
```

## 9. CASE STATEMENTS - Conditional Logic
**Intuition**: If-then-else logic in SQL.

```sql
-- Simple CASE
SELECT name,
       CASE 
           WHEN age < 18 THEN 'Minor'
           WHEN age BETWEEN 18 AND 64 THEN 'Adult'
           ELSE 'Senior'
       END as age_group
FROM users;

-- CASE in aggregations
SELECT department,
       COUNT(CASE WHEN salary > 100000 THEN 1 END) as high_earners,
       COUNT(CASE WHEN salary <= 100000 THEN 1 END) as regular_earners
FROM employees
GROUP BY department;
```

## 10. UNION & SET OPERATIONS - Combining Result Sets
**Intuition**: Stack or combine results from multiple queries.
**Rule**: Queries must have same number of columns with compatible types.

```sql
-- UNION: Combines and removes duplicates
SELECT name FROM customers
UNION
SELECT name FROM suppliers;

-- UNION ALL: Keeps duplicates (faster)
SELECT product_id FROM orders_2023
UNION ALL
SELECT product_id FROM orders_2024;

-- INTERSECT: Only rows in both results
SELECT customer_id FROM orders
INTERSECT
SELECT customer_id FROM returns;

-- EXCEPT: Rows in first but not second
SELECT customer_id FROM all_customers
EXCEPT
SELECT customer_id FROM churned_customers;
```

## 11. STRING & DATE FUNCTIONS
```sql
-- String functions
SELECT CONCAT(first_name, ' ', last_name) as full_name FROM users;
SELECT UPPER(name), LOWER(name), LENGTH(name) FROM products;
SELECT SUBSTRING(email, 1, POSITION('@' IN email) - 1) as username FROM users;

-- Date functions
SELECT CURRENT_DATE, CURRENT_TIMESTAMP;
SELECT DATE_PART('year', order_date) as year FROM orders;
SELECT DATE_TRUNC('month', created_at) as month FROM events;
SELECT order_date + INTERVAL '7 days' as expected_delivery FROM orders;
SELECT AGE(CURRENT_DATE, birth_date) as age FROM users;
```

## 12. ADVANCED PATTERNS

### Recursive CTEs
**Intuition**: Query hierarchical data (org charts, categories, etc.)

```sql
-- Employee hierarchy
WITH RECURSIVE org_chart AS (
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY level, name;
```

### Pivot (Rows to Columns)
```sql
SELECT 
    product,
    SUM(CASE WHEN quarter = 'Q1' THEN sales ELSE 0 END) as Q1,
    SUM(CASE WHEN quarter = 'Q2' THEN sales ELSE 0 END) as Q2,
    SUM(CASE WHEN quarter = 'Q3' THEN sales ELSE 0 END) as Q3,
    SUM(CASE WHEN quarter = 'Q4' THEN sales ELSE 0 END) as Q4
FROM quarterly_sales
GROUP BY product;
```

### Running Differences
```sql
SELECT order_date, total,
       total - LAG(total) OVER (ORDER BY order_date) as diff_from_prev
FROM orders;
```

### Cumulative Distinct Count
```sql
WITH numbered_orders AS (
    SELECT customer_id, order_date,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_num
    FROM orders
)
SELECT order_date,
       COUNT(CASE WHEN order_num = 1 THEN 1 END) as new_customers,
       COUNT(DISTINCT customer_id) as total_customers
FROM numbered_orders
GROUP BY order_date;
```

### Complex Filtering with EXISTS
```sql
-- Find customers who ordered ALL products in category 'electronics'
SELECT c.name
FROM customers c
WHERE NOT EXISTS (
    SELECT p.id
    FROM products p
    WHERE p.category = 'electronics'
    AND NOT EXISTS (
        SELECT 1
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        WHERE o.customer_id = c.id AND oi.product_id = p.id
    )
);
```

## 13. PERFORMANCE TIPS
```sql
-- Use indexes for WHERE, JOIN, ORDER BY columns
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_user ON orders(user_id, order_date);

-- Avoid SELECT * in production
-- Use EXPLAIN to analyze query plans
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 123;

-- Use WHERE before JOIN when possible
-- Limit subquery results early
-- Use EXISTS instead of IN for large subqueries
```

## 14. DATA MODIFICATION
```sql
-- INSERT
INSERT INTO users (name, email) VALUES ('John', 'john@email.com');
INSERT INTO users (name, email) SELECT name, email FROM temp_users;

-- UPDATE
UPDATE users SET status = 'active' WHERE last_login > CURRENT_DATE - 30;
UPDATE products SET price = price * 1.1 WHERE category = 'electronics';

-- DELETE
DELETE FROM logs WHERE created_at < CURRENT_DATE - INTERVAL '90 days';

-- UPSERT (INSERT or UPDATE if exists)
INSERT INTO users (id, name, email) VALUES (1, 'John', 'john@email.com')
ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, email = EXCLUDED.email;
```

---

## QUICK REFERENCE: Query Execution Order
1. FROM (including JOINs)
2. WHERE
3. GROUP BY
4. HAVING
5. SELECT
6. DISTINCT
7. ORDER BY
8. LIMIT/OFFSET

**Remember**: You can't use column aliases from SELECT in WHERE, but you can in ORDER BY and HAVING.
