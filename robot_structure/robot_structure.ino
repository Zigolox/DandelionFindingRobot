#include <Wire.h>
#include <VL53L1X.h>
#include <Servo.h>
#include <EEPROM.h>
#include "math.h"


#define DELTA 0.01f
#define pi 3.14159265f

//For motor controll:
const byte interrupt_pin = 0;
const byte encoder0pinA = 2;//A pin -> the interrupt pin 0
const byte encoder0pinB = 3;//B pin -> the digital pin 11
byte encoder0PinALast;
int duration = 0;//the number of the pulses
bool Direction;//the rotation direction
int measure_time = 200;

class cRobot{

public:

  cRobot(int L, int W, int R,int r) { //Construct robot

    len=L;
    width=W;
    r_0=R;
    r_wheel=r;
  }


  void stop_robot() { //Stop the robot
    setVelocity(0, 0);
  }

  void reverse(int vel){ //Go backward at constant velocity

    setVelocity(-vel, -vel);

  }

  void forward(int vel) { //Go forward at constant velocity
    setVelocity(vel, vel);

  }

  void rotate(int vel, char dir){
    if(dir = 'r'){
      setVelocity(vel,-vel);
    }
    if(dir = 'l'){
      setVelocity(-vel,vel);
    }
  }

  static void wheelSpeed()
    {
      int Lstate = digitalRead(encoder0pinA);
      if((encoder0PinALast == LOW) && Lstate==HIGH)
      {
        int val = digitalRead(encoder0pinB);
        if(val == LOW && Direction)
        {
          Direction = false; //Reverse
        }
        else if(val == HIGH && !Direction)
        {
          Direction = true;  //Forward
        }
      }
      encoder0PinALast = Lstate;

      if(!Direction)  duration++;
      else  duration--;
    }

  void writeToEEP(int index, char value) {
    EEPROM.write(index, value);
  }

  byte readFromEEP(int index) {
    return EEPROM.read(index);
  }

  unsigned int measureVelocity(){

    Direction = true;//default -> Forward
    pinMode(encoder0pinB,INPUT);
    attachInterrupt(interrupt_pin, wheelSpeed, CHANGE);

    delay(measure_time);

    //Serial.println(duration*100/measure_time);
    detachInterrupt(interrupt_pin);
    float temp = duration;
    duration = 0;

    return min(abs(temp*50/measure_time),255);
  } //Measure velocity of a motor

  unsigned int gatherVelocityData(unsigned int output) {//Used to take measurements
    unsigned int vel;
    forward(output);
    delay(550);
    vel = measureVelocity();
    Serial.print(vel);
    Serial.print("\t");
    Serial.println(output);

    return  vel;

  }

  void robot_attach(int servo_PIN);

  int calulateOutput(int vel){
    int out;
    bool sgn = vel>=0;
    vel = abs(vel);

    int index = 0;

    while(EEPROM.read(index) < vel and index < 255){
      index++;
    }

    out = EEPROM.read(index+256);

    if(sgn) {
      return out;
    }
    else {
      return -out;
    }

  } //Calculate output for certain velocity
  bool found_target() {  //Look for the target
    bool found = 1;         //True if target is found
    if(found) {
      target_angle = 0; //If found, show what angle object is at
    }
    return found; //Return found variable
  }

  bool go_to_target() {
    if(target_angle > min_angle) {
      rotate(v0,'l');
    }
    else if(target_angle < -min_angle) {
      rotate(v0,'r');
    }
    else {
      forward(v0);
    }
  }

private:


  void setVelocity(int v_left, int v_right) { //Private method to set velocity of engines

    int out_left = calulateOutput(v_left);
    int out_right = calulateOutput(v_right);

    /*
    Serial.print("Left v: ");
    Serial.print(v_left);
    Serial.print(" Right v: ");
    Serial.println(v_right);
    Serial.print("Left out: ");
    Serial.print(out_left);
    Serial.print(" Right out: ");
    Serial.println(out_right);
    */
    analogWrite(E2,abs(out_right));



    if(v_left<0)
    {
      digitalWrite(M2,LOW);
    }

    else
    {
      digitalWrite(M2,HIGH);
    }

    analogWrite (E1,abs(out_left));

    if(v_right<0)
    {
      digitalWrite(M1,LOW);
    }

    else
    {
      digitalWrite(M1,HIGH);
    }

  }

  void setOmega(float omega) { //Drive at velocity v0 and dtheta/dt = omega
    float Dv = omega*width/2;
    Dv = Dv/((float) vel_koeff);

    if(side) {setVelocity(v0+Dv, v0-Dv);}
    else {setVelocity(v0-Dv, v0+Dv);}


  }

  int E1 = 6;     //M1 Speed Control
  int E2 = 5;     //M2 Speed Control
  int M1 = 7;     //M1 Direction Control
  int M2 = 4;     //M2 Direction Control

  float len;      //length of robot (between lidar)
  float width;    //Width of robot (between wheels)
  float r_0;      //Desired distance from wall
  float r_wheel;  //Radius of wheels



  float y;  //Current distance from wall (-r_0)
  float theta;  //Offset angle
  bool side = false;  //False - follow left wall, True - follow right wall

  int v0 = 100; //Standard velocity of robot

  float k = 0.65; //Value in equation of motion. 1-5

  double T_100;

  float vel_koeff = 2*1.73; //Ratio between rad/s and velocity unit

  char DATA_velocity[200];  //Array with velocity data
  char Data_output[200];

  bool located = false; //Is the target located

  float target_angle;   //Angle (or some quantity related to that)
                        //between robot heading and target

  float min_angle = 10; // Minimum acceptable deviation


};


cRobot robot(160,250,95,45); //Create robot object
void setup() {

  Serial.begin(9600);
  Wire.begin();
  Wire.setClock(400000);
  Serial.println("MID");
  delay(1000);



}

void loop() {
  if(robot.found_target()) {
    robot.go_to_target();
  }
  else {
    robot.rotate(100,'r');
  }

}
