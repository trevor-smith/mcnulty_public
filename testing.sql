# What customers are from the UK?
SELECT * FROM Customers where Country = 'UK';

# What is the name of the customer who has the most orders?
select ContactName, count(*) as count
  FROM Customers
  JOIN Orders on (Customers.CustomerID = Orders.CustomerIDID)
  GROUP BY ContactName
  order by count desc;


CREATE TABLE reviews_raw
(
ROW_ID  SERIAL PRIMARY KEY
,DATA  TEXT
);


select player1 from us_men_2013
union
select player2 from us_men_2013
union
select player1 from french_men_2013
union
select player2 from french_men_2013
union
select player1 from aus_men_2013
union
select player2 from aus_men_2013
union
select player1 from wimbledon_men_2013
union
select player2 from wimbledon_men_2013
order by player1;
