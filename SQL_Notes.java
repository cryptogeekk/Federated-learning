

								NEW SQL COURSE NOTES

->Database is a collection of a data with an aim of manipulating it.
->Database mangement system is a interface which helps to manipilate a database.
->PostgresSQL,MySQL,oracle databse are not database but database management system.

->SQL VS MySQL
	SQL is a language we use to talk to our database.
		ex:-add a data,delete a data,finding a student who are greater than 18 years.

	MySQL is a database management system.
	MYSQL makes it special with SQL with the unique features MYSQL provides since it is a DBMS.

->When we start a SQL a database server is created and is running.Every database we create
	is stored inside that server.

->show databases;	//It will show all the databaes on the server.
->create database <name>;   //creates a databse in the server.
->drop database <name>;		//It will delete the database.
->use <databse_name>;		//It will bring a database into use.
->select database();		//It will show the currently using database.

->Tables 
	A database is a bunch of table.Table holds the data.
	Columns are the header of a table.Rows are the data inside it.

->Data types
	A column header should be specified with a data type at first.It prevents the inconsistency
	of a data.
		ex:-If name is defined as a char initially its all data  should contains char value
			its data cannot contain a integer value.

	Numeric type
		int,float,double,Bit 
	String types
		varchar,text,longtext
	Date types
		date,time,year

->Creating a table
create table <tablename>
	(
		column_name data_type,
		column_name data_type
		); 

->show tables;  //It shows the table present inside a database.
->show columns from <tablename>; or desc <tablename>  //describes the tables.
->drop table <tablename>;


->Inseritng into table
	insert into cats(name,age) values("jetson",7);

->Retrieving a data from table;
	select * from table;
	ex:- select * from cat;

->Inserting a multiple value in a table.
insert into cat(name,age)
    -> value("krishna",15),
    -> ("aditya",10),create table unique_cats(
			cat_id int not null auto_increment,
			name varchar(100),
			age int,
			primary key(cat_id));
    -> ("gopal",19)
    -> ;

->If there is a warning,Then we can know by 
	->show warnings;

->Null
	It is a value not known.
	name varchar(100) NOT NULL,
	age int not null 					   //not null helps to prevent from value being NULL

	->lets say we defined earlier into a table that a data should not be null and if we donot 
	assign a value to it then it will automatically assign the value 0.
		FOR ex:-
			a table contains name and age.if we only insert name then age will be automatically
			assigned 0.If tthere is a varhcar then the string will be empty.

->Default value
	age int default 19  //It is used to make a default of a variable which will be applied
						//to all the values created of that variable.

->Primary Key
	In a database there may be a data with a smiliar name. while retrieving it we may get confused.
	For this we define  a primary key which is unique for everyone.
		create table unique_cats(
			cat_id int not null,
			name varchar(100),
			age int,
			primary key(cat_id));

	Here if we are assinging the primary key as a cat_id then we must pass the unique value of 
	cat_id while inseritng the data. If we try to insert a duplicate key then it will shown
	an error mentioning a duplicate key.

	->We can make any variable as a primary key,not necessary that it should be a cat_id.

	->Auto_increment is a syntax to assign a unique value in a increment order to cat_id.
	By doing so we donot have to pass a unique id every time we insert data into the table.
		create table unique_cats(
			cat_id int not null auto_increment,
			name varchar(100),
			age int,
			primary key(cat_id));

->Foreign Key
	CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    PersonID int,
    PRIMARY KEY (OrderID),
    FOREIGN KEY (PersonID) REFERENCES Persons(PersonID)
);

	->The child table contains foreign key. The foregin key is established in child table with 
	the help of primary key in a parent table.

	->We should not make a foreign key by making first the element as a primary key and 
	then foreign key.
	

->CRUD(create,read,update,delete)
	select * from table;
		here * means to select all the columns presented in the table.

	->If we want a particular variable only from a table then we use a command
		select name from tables;	//gives all the name inside a table.

	->We can even select a multiple data from a table.
		select name,age from table;

	->Searching in a database.
		select * from cats where age=4; 	//this will search for all the cats whose age is 4
		select * from cats where name="egg";	//While searching for name the string we pass is
												//case insensitive.

->Aliases

	->We can display the data into heading name as we want.
		ex: lets say we have a two table of cat and dog.Both contains a name.While displaying 
			name it will be quite confusion which name is of which.SO we can display name of
			a cat as a cat_name and nameof a dog as name_dog.

			select name as 'cat_name', breed as 'kitty bread ' from cats;

->Update 
	Lets say we forgot the password and hit the reset button.Now new password will be asked 
	which should be updated with an old password in a database.

	->update cats set breed='shorthair'
	  where breed='tabby';

	 -> update cats set age=14 where name='misty';

	 ->Double update
	 	update shirts set color='off white',shirts_size='XS' where color='white';


->Delete 
	delete from cats where name='egg';
	Deleting the name='egg' only deletes the data not the whole table.

	delete from shirts;		//Deletes all the data from shirts.




ALTER TABLE Orders
ADD CONSTRAINT FK_PersonOrder
FOREIGN KEY (PersonID) REFERENCES Persons(PersonID);

->Deleting all the elements from the Tables
	->delete from shirts;

->Deleting the whole table
	->drop table shirts;


->Files
	When we type the sql commands then sometimes we may get an error.For this we save all
	the commands in a file and run it.The file in which we save our commands must be in a 
	same directory where our sql commands run.
		executing sql files
			source filename.sql

->String Functions
	When we read the data we read in its raw form i.e when we retrieve the data we say 
	select color,age from person but we donot add,combine,make uppercase/lowercase or perform
	ny operation on data.when we perform the operation it is done with the string function.

	->concat function
		concat(x,y,z)
		concat is used to combine two pieces of data.lets say we retrieve the first name 
		as well as last name but if we want to know the full name then we conact the first
		name and last name. 

			concat(column,anothercolumn)
			To add a retrievingquired text between two column
				concat(column,'text',anothercolumn,'more text')

				ex:-select concat(author_fname,' ',author_lname) from books;

				->Giving it a name
					select concat(author_fname,' ',author_lname) as full_name from books;

				->Obtaining first,last name and full name from table
					select author_fname as first,author_lname as last,
					concat(author_fname,' ',author_lname) as full from books;

	->Using a  seperator
		concat_ws()

		If we want to pass the - between every data we retrieve the instead of doing
			concat(author_fname,'-',author_lname,'-',page) as full from books;

			we can do
			concat_ws('-',title,author_fname,author_lname) from books;

->Substring function
	It allows us to only retriev the some portion of a string.In SQL index starts from 1.
	select substring('Hello world',1,4);
		result:Hell

	select substring('Hello world',7)
		It will give substring starting from index 7.i.e world 

	select substring('Hello world',-3)
		result:rld(It satrts from backward.)

	->select substring(title,1,10) from books;
		result:gives only the first 10 Character of a title.

	->concatenating with a Substring
		select concat(short_title,'.....') from books;
		select substring(title,1,10) from books as short_title

		select concat(substring(title,1,10),'.....') from books;


->Replace function
	select replace('Word','word to be removed','word to be placed')
	select replace('Hello world','Hell','Heaven');
	select replace('Hello world','l','7');
	select replace('Bring coffee milk',' ',' and ');
	select replace(title,'e','3') from books;

	->We can nest the Replace functions with other functions.
		ex:-select substring(replace(title,'e','3'),1,10) from books;
			->It first execute the replace command and then after it executes the substring
			command.

->Reverse Function
	Used to reverse the word.
	select reverse('meow meow');
	select reverse(author_fname) from books;

	Using replace function with concatenate function.
		select concat(author_fname,' ',reverse(author_fname)) from books;



->char_length functions
	It is used to count the length of a given character.
	select char_length('Hello world');
		result:11

	Using char_length function with another functions.
	 1.select author_lname,char_length(author_lname) from books;
	 2.select concat(author_lname,' is ',char_length(author_lname),' characters long');

	 ->We can use sql-format.com to format the queries.

->Upper and lower case letters
	select lower('HELLO WORLD');
	select upper(title) from books;


->Distinct
	When we say select author_fname from books; it will give all the author_fname. If there 
	is a duplicate values then it will be repeated. But if we use the distinct keyword
	then we only get distinct author name.
		ex:select distinct author_fname from books;

	->Distinct works for both varchar and int values.
	-- select distinct author_fname,author_lname from books;
	-- select distinct concat(author_fname,' ',author_lname) from books;

->Order By
	Order by is used to sort the data.
		syntax:select author_lname from books order by author_lname;

	->In default it sorts in ascending order but we can change it by;
		select author_lname from books order by author_lname desc;  //desc means descending.

	->It works for integer
		select released_year from books order by released_year;

		->select title,pages from books order by released_year;
			here we select title and pages but we are ordering by released_year.So the result
			will be ordered by released_year not by title or pages.It is not necessary that
			we have to orders the things by the data we select.

		->select title,author_fname,author_lname from books order by 2;
			Here 2 means we are ordering with respect to author_fname.(index number)

		->select author_fname,author_lname from books order by author_lname,author_fname;
			Lets say we have two name Harris Dan and Harris Freida. When we sort by author_lname
			then we sort harris but in first name Freida may come first then Dan. To have 
			a proper sorting we also sort by the first name also.Here sorting by author_fname 
			and author_lname comes handy.

		->select title,pages from books order by pages desc limit 1;
			limit 1 is used to select the the data at the 1st index. i.e the top most data.

->Like 
	Lets say the author name is dave but we only remember the name da. Now we can use Like
	keyword to search da and it will find any word starting from da onwards.This is a better
	way of searching.
		syntax:select author_fname from books where author_fname like '%da%';
			it will search for anything between % symbol.

			'da%' : starts with da and then anything afterwards.
			'%da' : must be ending with da before da nything can be there.
				The % sign is called wild card character.

		->When we search in the reddit or amazon in the backend the like is being implemented.


		->'___' 
			here there is 3 underscores number . 3 underscore means three digit number.

		syntax:select stock_quantity from books where stock_quantity like '__'
			double underscore will give stock_quantity which is two digit long.

		->But what if we are searching for the % and _ character in a book.
			Then we use \ sign.
				syntax: '%\%%' and '%\_%'

--------------------------------------Aggregate Functions-----------------------------------------------------
->Count
	It is used to count the data in a given dataset.
		ex:-
			select count(*) from books;
			select count(title) from books where title like '%the%';
->Limit
	limit is used to select number of rows
		limit 1 //select only one row.
		limit 10 //select 10 rows
		select * from books order by pages asc limit 1;

->Group By
	The GROUP BY statement groups rows that have the same values into summary rows,
	 like "find the number of customers in each country".

	 ->It is also used to select the identical data from the database.
	 	select author_fname from books group by author_fname;

	 		here we only select the non-repeated author_fname data.

	 ->If we want to find which author wrote how much book then we can find with the 
	 	help of aggregate function and count function.
	 		select author_lname,count(*) from books group by author_lname;

	 ->If we want to know in which year how much book released then
	 	select released_year,count(*) from books group by released_year;

->Max and Min 
	syntax:select max(pages) from books;

	->select max(pages),title from books;
		When we do this we get an error. This is because the max(pages) and title are independent
		of each other.It gives the max page but gives the first title.

	->Sub queries:
		select * from books 
			where pages=(select min(pages) from books);

		In this tpype of querie first select min(pages) from books is executed and then on
			the basis of it title is selected.

		->Next method 
			select * from books order by pages asc limit 1;

->Using Min and Max function with group by:
	->Find the year in which each author published their first book.
		select author_fname,min(released_year) from books group by author_fname,author_lname;

		Note:When using group by function we can only display the things on the basis of which
			we have sorted.here since we have only sort on the basis of author_lname,author_fname
			we can only display author_fname and author_lname but we cannot display title.
			But for displaying title we can use concat functions.



	->We can also use group function with concat.
		select concat(author_fname,' ',author_lname) as author_name,max(pages) as pages from
		 books group by author_fname,author_lname;

	->Sum 
		It is used to sum all the elements.
			ex:Summation of salary,summ of pages.

			synatx:select sum(pages) from books;

	->select author_fname,author_lname,sum(pages) from books group by author_fname,author_lname;


	->Average
		It is used to find the average.	
			synatx: select avg(released_year) from books;

			->select author_fname,author_lname,avg(pages) from books group by author_fname,
				author_lname;

	->Note
		When we are using two times as aliases in a query then in second query we have to 
		quote the as in ''
			ex: select released_year as year,count(*) as 'no.of books',avg(pages) as 'avg pages' from books group by released_year;
 
--------------------------------------Refifnig Section-------------------------------------------------------
->Char has a fixed length.
	If we assign title a char(5) then all the title in a database must be of length five.
	if it exceeds more than 5 then it will be truncated and if it is less than 5 then 
	addition of spaces will make it character of 5.

	->The length can be in between 0 to 255.
	->When char value is retrieved the spaces which are given to fill the length are eliminated.
	->char is faster for fixed length text. we use char when we are confident that the given 
		chracter will be of fixed length. 

->Varhcar
	Varhcar is of variable length.
	->When we are not sure about the length.we use it. 
	->var char is more memory efficient.

->Decimal
	Decimal(5,2)
		It is 5 digits long with two digit after the decimal point.
		It is used to store the decimal point value.

		-- create table items(
		--     price decimal(5,2)
		-- );

		insert into items(price) values(52);
		insert into items(price) values(52.5);

		If we insert the value higher than the decimal specified then it will result
		some unexpected values.


-----------------------------------Logical Operator-------------------------------------------

->Not equal(!=)
	 select title from books where released_year !=2017;

->Not like 
	It is opposite of like. It will find all the elements not containing specified under ''.
		select title from books where title not like '%w%';

->Greater than 
	It selects all the element which is greater than certain specified in ''.
		select * from books where released_year >2000;
		select * from books where released_year >=2000;
		Here while doing comapring it returns a boolean value true or false.On the basis
		of that result is returned.

->Boolean Logic
	->select 99>1;
		It will return 1. 1 is used to say True and 0 for false.

->select 'A' < 'a';
	In my SQL upper case (A) is equal to lower case (a).

->	select title,author_lname from books where author_lname='eggers';
	select title,author_lname from books where author_lname='Eggers';

		Above two statements will give same result since in sql e is equal to E.
			select 'a'<='H';	result 1.
			Here since it is case insensitive the above statement will ve equuivalent to
			select 'a'<='h'; and since 'a' is less than 'h' it is returning 1.

->Less than
	It is just opposite of greater than.
	synatx:select title,pages from books where pages<600; 


->Second normal forma
	For a table to be in 2nd normal form it should be in first normal form first.
	Partial depenedcy should not be there.
	Dependency is when one tuples depend upon other two tuples. Partial depenedcy
	is when it depends upon one.ex professor_name is only depending upon subject.

	 SO we can remove it either by moving the teacher_name
	to another table and assign every row a primary key .

->Third normal form
	Composite key:Combination of two primary key.
	Transitive depenedcy:When there is a attribute which donot depend upon 
		the primary key rather on other attribute.

	Solution to the Transitive depenedcy.
		Put the attribute1 and attribute2(depend upon attribute1) in the new table.

->BCDF
	->Should be in third normal form.
	->Prim

->Fourth Nomral form.
	->It should be in BCNF
	->There should be no multi value depenedcy.

		Multi value depenedcy:For single value of A more than two value exists.
			i.e B1 and B2.
			->For a column to have multi value depenedcy it should have 3 columns.
			->For this table with A,B,C columns, B and C should be independent.

		There can be a table in which same s_id have different course and different
		hobby. This leads to the arise of same two rows again leading to a multi value 
		depenedcy. To resolve this problem we can seperate s_id,course
		and s_id,hobby into two seperate tables.


->Functional Dependency
	A functional dependency A->B in a relation holds if two tuples having same value
	of attribute A also have same value for attribute B. For Example, in relation 
	STUDENT shown in table 1, Functional Dependencies.

->No loss decomposition
	If we decompose a relation R into relations R1 and R2,
		Decomposition is lossy if R1 ⋈ R2 ⊃ R
		Decomposition is lossless if R1 ⋈ R2 = R

->Dependency Preserving Decomposition
	->If we decompose a relation R into relations R1 and R2, All dependencies of R
	either must be a part of R1 or R2 or must be derivable from combination of FD’s
	 of R1 and R2.
	->For Example, A relation R (A, B, C, D) with FD set{A->BC} is decomposed into
	 R1(ABC) and R2(AD) which is dependency preserving because FD A->BC is a part of
	 R1(ABC).

Super Key:
->Super Key is an attribute (or set of attributes) that is used to uniquely
 identifies all attributes in a relation. All super keys can’t be candidate keys
  but its reverse is true. In a relation, number of super keys are more than number 
  of candidate keys.

Example:
We have a given relation R(A, B, C, D, E, F) and we shall check for being super 
keys by following given dependencies:

Functional dependencies         Super key
AB->CDEF                         YES
CD->AEF                          YES
CB->DF                           NO
D->BC                            NO 
By Using key AB we can identify rest of the attributes (CDEF) of the table. 
Similarly Key CD. But, by using key CB we can only identifies D and F not A and E.
Similarly key D.

->Candidate Key
	
Candidate key is a set of attributes (or attribute) which uniquely identify the 
tuples in a relation or table. As we know that Primary key is a minimal super key, 
so there is one and only one primary key in any relation but there is more than one
candidate key can take place. Candidate key’s attributes can contain NULL value
 which oppose to the primary key.
Example:

Student{ID, First_name, Last_name, Age, Sex, Phone_no} 
Here we can see the two candidate keys ID and {First_name, Last_name, DOB, Phone_no}.
 So here, there are present more than one candidate keys, which can uniquely
  identifies a tuple in a relation.



select player_name from player where id_no=(select count(*) from player group by id_no limit 1 );


 


 insert into customer(cno,cname,street,zip,phone) value(1,'gopal','kkr',580,9844840763), (2,'krishna','npj',620,9896183709), (3,'Rajesh','ktm',780,123456789), (4,'gopal','kkr',580,9844840763);


insert into employees(eno,ename,zip,hdate) value(1,'abishek',012,28-10-2018), (2,'dhruv',345,27-9-2017);
